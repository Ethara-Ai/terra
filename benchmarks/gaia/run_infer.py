import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, List, Sequence, cast

import huggingface_hub
import pandas as pd
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from PIL import Image

from benchmarks.gaia.config import INFER_DEFAULTS
from benchmarks.gaia.scorer import question_scorer
from benchmarks.gaia.utils import image_to_jpg_base64_url, image_to_png_base64_url
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
from benchmarks.utils.agent_context import create_agent_context
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.console_logging import summarize_instance
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import create_docker_workspace, remote_image_exists
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from benchmarks.utils.version import IMAGE_TAG_PREFIX
from openhands.sdk import (
    Agent,
    Conversation,
    Event,
    ImageContent,
    Message,
    MessageEvent,
    TextContent,
    Tool,
    get_logger,
)
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import APIRemoteWorkspace


load_dotenv()

logger = get_logger(__name__)

# Cache directory for GAIA dataset files
DATASET_CACHE_DIR = Path(__file__).parent / "data"


def _resolve_file_path(split: str, file_name: str) -> Path:
    """Resolve attachment file path, checking nokor/ then 2023/ directories."""
    nokor_path = DATASET_CACHE_DIR / "nokor" / split / file_name
    if nokor_path.exists():
        return nokor_path
    return DATASET_CACHE_DIR / "2023" / split / file_name


class GAIAEvaluation(Evaluation):
    """
    GAIA benchmark evaluation implemented as a child of the
    abstract Evaluation orchestrator.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def prepare_instances(self) -> List[EvalInstance]:
        """Load GAIA dataset from HuggingFace and prepare instances."""
        logger.info("Setting up GAIA evaluation data")

        # Load dataset from HuggingFace
        assert self.metadata.details is not None
        level = self.metadata.details.get("level")
        if not level:
            raise ValueError(
                "GAIA level must be specified in metadata.details['level']"
            )

        logger.info(
            f"Loading GAIA dataset: {level}, split: {self.metadata.dataset_split}"
        )

        # Check for local dataset path (CLI arg, env var, or local nokor directory)
        local_dataset_path = None
        if self.metadata.details and self.metadata.details.get("dataset_file"):
            local_dataset_path = str(
                Path(self.metadata.details["dataset_file"]).resolve()
            )
        if not local_dataset_path:
            local_dataset_path = os.getenv("GAIA_LOCAL_DATASET_PATH")

        if local_dataset_path:
            # Load from local JSONL file
            logger.info(f"Loading local dataset from: {local_dataset_path}")
            df = pd.read_json(local_dataset_path, lines=True)
        else:
            # Fallback to HuggingFace
            dataset = cast(DatasetDict, load_dataset("gaia-benchmark/GAIA", level))

            # Download dataset files
            logger.info(f"Downloading GAIA dataset files to {DATASET_CACHE_DIR}")
            DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            huggingface_hub.snapshot_download(
                "gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir=str(DATASET_CACHE_DIR),
            )

            # Convert to pandas
            df = cast(pd.DataFrame, dataset[self.metadata.dataset_split].to_pandas())

        df.rename(columns={"task_id": "instance_id"}, inplace=True)

        # Filter completed instances
        completed_instances = self._get_completed_instances()
        if completed_instances:
            df = cast(
                pd.DataFrame, df[~df["instance_id"].isin(list(completed_instances))]
            )
            logger.info(f"Filtered out {len(completed_instances)} completed instances")

        # Filter by selected_instances_file if provided (before applying eval_limit)
        if self.metadata.selected_instances_file:
            with open(self.metadata.selected_instances_file, "r") as f:
                selected_ids = set(line.strip() for line in f if line.strip())

            before_selection = len(df)
            df = cast(pd.DataFrame, df[df["instance_id"].isin(list(selected_ids))])
            logger.info(
                "Filtered to %d selected instances from file (from %d)",
                len(df),
                before_selection,
            )

            if len(df) == 0:
                logger.warning(
                    "Selected instances file %s produced 0 matching instances",
                    self.metadata.selected_instances_file,
                )

            # Keep all requested IDs; ignore eval_limit when selections are provided
            self.metadata.eval_limit = len(df)

        # Apply eval_limit if specified (only when no explicit selection)
        elif self.metadata.eval_limit and self.metadata.eval_limit > 0:
            df = cast(pd.DataFrame, df.head(self.metadata.eval_limit))
            logger.info(f"Limited to {len(df)} instances due to eval_limit")

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info(f"Total instances to process: {len(instances)}")
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create workspace and copy necessary files.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
            forward_env: Environment variables to forward into the workspace.
        """
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        logger.info(f"Preparing workspace for instance {instance.id}")

        if self.metadata.workspace_type == "docker":
            # Use linux/arm64 on Apple Silicon Macs for native performance
            import platform as _platform

            docker_platform = (
                "linux/arm64" if _platform.machine() == "arm64" else "linux/amd64"
            )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-gaia-binary"
            )
            workspace = create_docker_workspace(
                agent_server_image=agent_server_image,
                base_image="nikolaik/python-nodejs:python3.12-nodejs22",
                build_target="binary",
                forward_env=forward_env,
                platform=docker_platform,
            )
        elif self.metadata.workspace_type == "remote":
            # For workflow, use APIRemoteWorkspace with pre-built GAIA image
            # GAIA uses a universal agent server image (one image for all instances)
            # Built from nikolaik/python-nodejs:python3.12-nodejs22 base
            # Using binary target (not binary-minimal) to include Chromium for browser operations
            # Image includes pre-cached MCP server to eliminate startup delays
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote workspace"
                )

            agent_server_image = (
                f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-gaia-binary"
            )

            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in container registry. "
                    f"Run 'benchmarks/gaia/build_images.py --push' to build and push it first."
                )

            logger.info(
                f"Using remote workspace with GAIA image {agent_server_image} "
                f"(tag prefix: {IMAGE_TAG_PREFIX}, resource_factor: {resource_factor})"
            )
            startup_timeout = float(os.getenv("REMOTE_RUNTIME_STARTUP_TIMEOUT", "600"))
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", "https://runtime.eval.all-hands.dev"
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
                target_type="binary",  # GAIA images use binary target
                forward_env=forward_env or [],
                resource_factor=resource_factor,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        # Create workspace directory
        workspace.execute_command("mkdir -p /workspace")

        # Handle file if present
        file_name = instance.data.get("file_name", "")
        if file_name:
            logger.info(f"Handling file: {file_name}")
            assert self.metadata.details is not None

            # Construct source file path
            src_file = _resolve_file_path(self.metadata.dataset_split, file_name)

            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
            else:
                extension_name = file_name.split(".")[-1].lower()

                # Skip images (jpg, png) - they'll be passed as base64 URLs
                if extension_name in ["jpg", "png", "jpeg"]:
                    logger.info(
                        f"Skipping image file {file_name} (will be passed as URL)"
                    )
                elif extension_name == "zip":
                    # Extract zip files
                    logger.info(f"Extracting zip file {file_name}")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(src_file, "r") as zip_ref:
                            zip_ref.extractall(temp_dir)
                        # Copy all extracted files to workspace
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                local_path = os.path.join(root, file)
                                workspace.file_upload(local_path, f"/workspace/{file}")
                else:
                    # Copy other files
                    logger.info(f"Copying file {file_name} to workspace")
                    workspace.file_upload(
                        str(src_file), f"/workspace/file.{extension_name}"
                    )

        # Install ffmpeg (needed for audio/video GAIA tasks)
        logger.info("Installing ffmpeg...")
        result = workspace.execute_command(
            "sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock"
            " && sudo dpkg --configure -a"
            " && sudo apt-get update -qq"
            " && sudo apt-get install -y -qq --no-install-recommends ffmpeg",
            timeout=300,
        )
        if result.exit_code != 0:
            logger.warning(f"Failed to install ffmpeg: {result.stderr}")
            logger.warning("Audio/video tasks may fail without ffmpeg")
        else:
            logger.info("✓ FFmpeg installed successfully")

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Run agent on a single GAIA instance and evaluate the result.
        """
        logger.info(f"Evaluating instance {instance.id}")

        # Build instruction
        instruction = self._build_instruction(instance)

        # Handle image URLs if the file is an image
        image_urls = []
        file_name = instance.data.get("file_name", "")
        if file_name:
            extension_name = file_name.split(".")[-1].lower()
            if extension_name in ["jpg", "png", "jpeg"]:
                # Load image and encode as base64
                assert self.metadata.details is not None
                src_file = _resolve_file_path(self.metadata.dataset_split, file_name)
                if src_file.exists():
                    image = Image.open(src_file)
                    if extension_name in ["jpg", "jpeg"]:
                        image_urls.append(image_to_jpg_base64_url(image))
                    else:
                        image_urls.append(image_to_png_base64_url(image))

        # Create agent
        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            agent_llm = build_eval_llm(self.metadata.llm)
            tools = get_default_tools(enable_browser=True)
            if self.metadata.enable_delegation:
                tools.append(Tool(name=DelegateTool.name))
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            assert tavily_api_key, "TAVILY_API_KEY environment variable is not set"
            condenser = None
            if self.metadata.enable_condenser:
                condenser = LLMSummarizingCondenser(
                    llm=build_eval_llm(self.metadata.llm, usage_id="condenser"),
                    max_size=self.metadata.condenser_max_size,
                    keep_first=self.metadata.condenser_keep_first,
                )
            # Load public skills (respects EXTENSIONS_REF env var)
            agent_context = create_agent_context()

            agent = Agent(
                llm=agent_llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
                agent_context=agent_context,
                condenser=condenser,
                mcp_config={
                    "mcpServers": {
                        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
                        "tavily": {
                            "command": "npx",
                            "args": ["-y", "tavily-mcp@0.2.1"],
                            "env": {"TAVILY_API_KEY": tavily_api_key},
                        },
                    }
                },
            )

        setup_acp_workspace(self.metadata.agent_type, workspace)

        # Create conversation

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
            delete_on_close=True,
        )

        # Send message and run
        with workspace_keepalive(self.metadata.agent_type, workspace):
            if image_urls:
                msg = Message(
                    role="user",
                    content=[
                        TextContent(text=instruction),
                        ImageContent(image_urls=image_urls),
                    ],
                )
                conversation.send_message(msg)
            else:
                conversation.send_message(instruction)
            # Run conversation with fake user responses to handle agent messages
            run_conversation_with_fake_user_response(conversation)

        # Extract answer from conversation history
        model_answer_raw = self._extract_answer_from_history(conversation.state.events)
        model_answer = self._parse_solution_tag(model_answer_raw)

        # Score the answer
        ground_truth = instance.data.get("Final answer", "")
        score = question_scorer(model_answer, ground_truth)

        logger.info(
            f"Instance {instance.id}: score={score}, "
            f"model_answer='{model_answer}', ground_truth='{ground_truth}'"
        )

        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            logger=logger,
        )

        # Collect history

        # Build test_result with optional ACP agent metadata
        test_result_data: dict[str, Any] = {
            "score": score,
            "model_answer_raw": model_answer_raw,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
        }
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(test_result_data, conversation)

        # Return evaluation output
        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result_data,
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
            instance=instance.data,
        )

    def _build_instruction(self, instance: EvalInstance) -> str:
        """Build GAIA-specific instruction for the agent."""
        question = instance.data.get("Question", "")
        file_name = instance.data.get("file_name", "")

        if is_acp_agent(self.metadata.agent_type):
            # ACP agents (Claude Code, Codex) may refuse prompts with
            # coercive/threatening language. Use a neutral variant that
            # conveys the same intent without triggering safety filters.
            instruction = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: you have access to all the relevant tools to solve it and the correct answer is findable with the tools available to you. Please do not respond with 'I cannot answer' or 'None found' — instead, keep exploring different approaches until you find the answer.
You MUST strictly follow the task-specific formatting instructions for your final answer.
Here is the task:
{task_question}
""".format(  # noqa: E501
                task_question=question,
            )
        else:
            instruction = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
You must make sure you find the correct answer! You MUST strictly follow the task-specific formatting instructions for your final answer.
Here is the task:
{task_question}
""".format(  # noqa: E501
                task_question=question,
            )

        # Add file information if present
        if file_name:
            extension_name = file_name.split(".")[-1].lower()
            if extension_name == "zip":
                # List files from zip
                src_file = _resolve_file_path(self.metadata.dataset_split, file_name)
                if src_file.exists():
                    with zipfile.ZipFile(src_file, "r") as zip_ref:
                        filenames = [f"/workspace/{f}" for f in zip_ref.namelist()]
                    filenames_str = ", ".join(filenames)
                    instruction += f"To solve this task you will have to use the attached files provided in the workspace at locations: {filenames_str}\n\n"  # noqa: E501
            elif extension_name in ["jpg", "png", "jpeg"]:
                instruction += "Image: To solve this task you will have to use the image shown below.\n\n"  # noqa: E501
            else:
                instruction += f"To solve this task you will have to use the attached file provided in the workspace at location: /workspace/file.{extension_name}\n\n"  # noqa: E501

        # Add GAIA-specific instructions
        instruction += """IMPORTANT: When seeking information from a website, REFRAIN from arbitrary URL navigation. You should utilize the designated search engine tool with precise keywords to obtain relevant URLs or use the specific website's search interface. DO NOT navigate directly to specific URLs as they may not exist.

For example: if you want to search for a research paper on Arxiv, either use the search engine tool with specific keywords or navigate to arxiv.org and then use its interface.
"""  # noqa: E501
        instruction += "IMPORTANT: You should NEVER ask for Human Help.\n"
        instruction += "IMPORTANT: Please encapsulate your final answer (answer ONLY) within <solution> and </solution> and report it back to users via a message, instead of the 'finish' tool. Your answer will be evaluated using string matching approaches so it important that you STRICTLY adhere to the output formatting instructions specified in the task (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)\n"  # noqa: E501
        instruction += (
            "For example: The answer to the question is <solution> 42 </solution>.\n"
        )
        instruction += "IMPORTANT: Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, express it numerically (i.e., with digits rather than words), do not use commas, and do not include units such as $ or percent signs unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities). If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"  # noqa: E501

        return instruction

    def _extract_answer_from_history(self, events: Sequence[Event]) -> str:
        """Extract the last agent message/thought from conversation history.

        This method searches for agent output from either MessageEvent or FinishAction.

        Note: When using RemoteConversation (with DockerWorkspace), there's a race
          condition where the final MessageEvent might not appear in the events list
          immediately after run() completes, due to WebSocket event streaming.
          This method implements a retry mechanism to handle that case.
        """
        # FIXME: Implement a more robust event synchronization mechanism in the SDK
        max_retries = 30  # Increased from 10 for better reliability
        retry_delay = 1.0  # Increased from 0.5s for slower networks
        retry_backoff = 1.2  # Exponential backoff factor

        # Log event type distribution for debugging
        if events:
            event_types = {}
            agent_events_count = 0
            for event in events:
                event_type = type(event).__name__
                event_types[event_type] = event_types.get(event_type, 0) + 1
                if hasattr(event, "source") and event.source == "agent":
                    agent_events_count += 1
            logger.info(
                f"Event type distribution: {event_types}, "
                f"agent-sourced events: {agent_events_count}"
            )

        for attempt in range(max_retries):
            # Search backwards through events for agent output
            if attempt == 0:
                logger.info(f"Extracting answer from {len(events)} events")
            else:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries}: searching for agent "
                    f"message in {len(events)} events"
                )

            for event in reversed(events):
                if not hasattr(event, "source") or event.source != "agent":
                    continue

                # Extract text from either MessageEvent or FinishAction
                text = None
                if isinstance(event, MessageEvent):
                    if event.llm_message and event.llm_message.content:
                        content = event.llm_message.content[0]
                        assert isinstance(content, TextContent)
                        text = content.text
                elif isinstance(event, ActionEvent) and isinstance(
                    event.action, FinishAction
                ):
                    text = event.action.message

                if text:
                    logger.info(
                        f"Found agent output on attempt {attempt + 1}: "
                        f"{type(event).__name__} - {text[:100]}..."
                    )
                    return text

            # If not found and we have retries left, wait and try again
            if attempt < max_retries - 1:
                current_delay = retry_delay * (retry_backoff**attempt)
                current_delay = min(current_delay, 5.0)  # Cap at 5 seconds
                logger.warning(
                    "Agent MessageEvent or FinishAction not found yet, "
                    f"waiting {current_delay:.1f}s before retry..."
                )
                time.sleep(current_delay)
                # Note: events is a reference to the conversation's events list,
                # which gets updated by the WebSocket callback in the background
            else:
                logger.error(
                    f"Could not find agent output after {max_retries} attempts "
                    f"and {sum(retry_delay * (retry_backoff**i) for i in range(max_retries)):.1f}s total wait time"
                )
                logger.error(
                    f"Final event types (last 10): "
                    f"{[type(e).__name__ for e in events[-10:]]}"
                )
                # Log more details about the last few events
                for event in events[-5:]:
                    logger.debug(
                        f"Event: {type(event).__name__}, "
                        f"source: {getattr(event, 'source', 'N/A')}, "
                        f"has content: {hasattr(event, 'llm_message')}"
                    )

        return ""

    def _parse_solution_tag(self, text: str) -> str:
        """Parse solution from <solution>...</solution> tags."""
        matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Return last match
        else:
            logger.warning(f"No <solution> tag found in: {text[:200]}...")
            return text  # Return raw text as fallback


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name for use as a directory name."""
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


def _get_model_folder_name(llm: "Any") -> str:
    """Get a clean folder name from the LLM config.

    Uses model_canonical_name if available, otherwise falls back to model field.
    """
    name = getattr(llm, "model_canonical_name", None) or llm.model
    return _sanitize_model_name(name)


def _reorganize_outputs(
    flat_output_dir: str,
    final_base_dir: str,
    model_folder: str,
) -> list[str]:
    """Reorganize flat evaluation outputs into per-instance/per-model structure.

    Reads the output JSONL files and conversation archives from flat_output_dir,
    then copies them into:
        final_base_dir/{instance_id}/{model_folder}/

    Returns list of reorganized instance directories.
    """
    reorganized: list[str] = []

    output_file = os.path.join(flat_output_dir, "output.jsonl")
    if not os.path.exists(output_file):
        logger.warning(f"No output.jsonl found in {flat_output_dir}")
        return reorganized

    instance_results: dict[str, list[dict]] = {}
    for fname in sorted(os.listdir(flat_output_dir)):
        if fname.startswith("output.critic_attempt_") and fname.endswith(".jsonl"):
            fpath = os.path.join(flat_output_dir, fname)
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        iid = data.get("instance_id", "unknown")
                        instance_results.setdefault(iid, []).append(
                            {"attempt_file": fname, "data": data}
                        )

    final_results: dict[str, dict] = {}
    with open(output_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                iid = data.get("instance_id", "unknown")
                final_results[iid] = data

    for instance_id in set(list(instance_results.keys()) + list(final_results.keys())):
        instance_dir = os.path.join(final_base_dir, instance_id, model_folder)
        os.makedirs(instance_dir, exist_ok=True)

        if instance_id in final_results:
            out_path = os.path.join(instance_dir, "output.jsonl")
            with open(out_path, "w") as f:
                f.write(json.dumps(final_results[instance_id]) + "\n")

        if instance_id in instance_results:
            for entry in instance_results[instance_id]:
                attempt_path = os.path.join(instance_dir, entry["attempt_file"])
                with open(attempt_path, "a") as f:
                    f.write(json.dumps(entry["data"]) + "\n")

        conv_archive = os.path.join(
            flat_output_dir, "conversations", f"{instance_id}.tar.gz"
        )
        if os.path.exists(conv_archive):
            conv_dir = os.path.join(instance_dir, "conversations")
            os.makedirs(conv_dir, exist_ok=True)
            shutil.copy2(conv_archive, os.path.join(conv_dir, f"{instance_id}.tar.gz"))

        for log_suffix in [".log", ".output.log"]:
            log_src = os.path.join(
                flat_output_dir, "logs", f"instance_{instance_id}{log_suffix}"
            )
            if os.path.exists(log_src):
                log_dir = os.path.join(instance_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                shutil.copy2(
                    log_src,
                    os.path.join(log_dir, f"instance_{instance_id}{log_suffix}"),
                )

        summary = _generate_instance_summary(
            instance_id, instance_dir, conv_archive, final_results.get(instance_id)
        )
        if summary:
            with open(os.path.join(instance_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        metadata_src = os.path.join(flat_output_dir, "metadata.json")
        if os.path.exists(metadata_src):
            with open(metadata_src, "r") as f:
                meta = json.loads(f.read())
            meta["eval_output_dir"] = os.path.join(
                "eval_outputs", instance_id, model_folder
            )
            with open(os.path.join(instance_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

        reorganized.append(instance_dir)

    return reorganized


def _generate_instance_summary(
    instance_id: str,
    instance_dir: str,
    conv_archive_path: str,
    final_result: dict | None,
) -> dict | None:
    """Generate a summary.json for a single instance with cost/tokens/latency/score."""
    import tarfile

    summary: dict = {
        "instance_id": instance_id,
        "score": None,
        "model_answer": None,
        "ground_truth": None,
        "cost": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "llm_calls": None,
        "avg_latency_seconds": None,
    }

    if final_result:
        test_result = final_result.get("test_result", {})
        summary["score"] = test_result.get("score")
        summary["model_answer"] = test_result.get("model_answer")
        summary["ground_truth"] = test_result.get("ground_truth")

    if os.path.exists(conv_archive_path):
        try:
            with tarfile.open(conv_archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("base_state.json"):
                        f = tar.extractfile(member)
                        if f:
                            state = json.loads(f.read().decode("utf-8"))
                            stats = state.get("stats", {})
                            usage = stats.get("usage_to_metrics", {})
                            for _key, metrics in usage.items():
                                summary["cost"] = metrics.get("accumulated_cost")
                                token_usage = metrics.get("accumulated_token_usage", {})
                                summary["prompt_tokens"] = token_usage.get(
                                    "prompt_tokens"
                                )
                                summary["completion_tokens"] = token_usage.get(
                                    "completion_tokens"
                                )
                                latencies = metrics.get("response_latencies", [])
                                summary["llm_calls"] = len(latencies)
                                if latencies:
                                    total_lat = sum(
                                        r.get("latency", 0) for r in latencies
                                    )
                                    summary["avg_latency_seconds"] = round(
                                        total_lat / len(latencies), 2
                                    )
                                break
                            break
        except Exception as e:
            logger.warning(f"Could not extract metrics from conversation archive: {e}")

    return summary


def main() -> None:
    """Main entry point for GAIA evaluation."""
    # Ensure AWS credentials file is not used - we authenticate via api_key (Bearer token)
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/dev/null"

    parser = get_parser()
    parser.add_argument(
        "--level",
        type=str,
        help="GAIA level to evaluate (e.g., 2023_level1, 2023_level2, 2023_level3, 2023_all)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Path to local JSONL dataset file (e.g., data/nokor/validation/custom_gaia_20.jsonl)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        help="Run a single instance by task_id (e.g., nokor_001)",
    )
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    instance_select_file = None
    if args.instance:
        instance_select_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        instance_select_file.write(args.instance + "\n")
        instance_select_file.close()
        args.select = instance_select_file.name
        args.n_limit = 1

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")

    # Validate arguments
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    model_folder = _get_model_folder_name(llm)

    # Construct dataset description
    dataset_description = f"{args.level}-{args.split}"

    # Construct output directory (use .tmp for flat output, then reorganize)
    structured_output_dir = construct_eval_output_dir(
        base_dir=os.path.join(args.output_dir, ".tmp"),
        dataset_name=dataset_description,
        model_name=model_folder,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create metadata
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "level": args.level,
            "dataset_file": getattr(args, "dataset_file", None),
        },
        eval_limit=args.n_limit,
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
    )

    # Create evaluator
    evaluator = GAIAEvaluation(metadata=metadata, num_workers=args.num_workers)

    # Run evaluation
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    # Reorganize outputs into per-instance structure
    logger.info("Reorganizing outputs into per-instance structure...")
    final_base_dir = args.output_dir
    reorganized = _reorganize_outputs(
        structured_output_dir, final_base_dir, model_folder
    )
    logger.info(f"Reorganized {len(reorganized)} instance outputs to {final_base_dir}")

    # Clean up temporary flat output
    tmp_dir = os.path.join(args.output_dir, ".tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    if instance_select_file:
        os.unlink(instance_select_file.name)

    for inst_dir in reorganized:
        rel_path = os.path.relpath(inst_dir, start=os.getcwd())
        logger.info(f"  {rel_path}/")

    logger.info("Evaluation completed!")
    output_paths = [os.path.relpath(d, start=os.getcwd()) for d in reorganized]
    print(json.dumps({"output_dirs": output_paths}))


if __name__ == "__main__":
    main()
