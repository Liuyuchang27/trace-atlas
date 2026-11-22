import argparse
import logging

from trace_atlas.agent import init_agent
from trace_atlas.atlas_profile import profile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    parser = argparse.ArgumentParser(description="TRACE_ATLAS Terminal CLI")
    parser.add_argument("--lang", default="cn", type=str, help="Language (en/cn)")
    parser.add_argument(
        "--llm-provider", default=None, type=str, help="Override the DeepSeek model provider"
    )
    parser.add_argument("--llm-model", default=None, type=str, help="LLM model identifier")
    parser.add_argument("--llm-api-key", default=None, type=str, help="LLM API key")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    options = profile(web_fallback=True)
    if args.llm_provider:
        options["llm_provider"] = args.llm_provider
    if args.llm_model or args.llm_provider:
        for role in ("searcher_model", "cognitive_model", "reader_model", "prefilter_model", "writer_model"):
            options[role] = args.llm_model or None
        if args.llm_model:
            options["llm_model"] = args.llm_model
    print(f"Initializing Trace Atlas ({options['llm_provider']}: {options['llm_model']})...")

    try:
        agent = init_agent(**options, llm_api_key=args.llm_api_key, lang=args.lang)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    print("TRACE_ATLAS Agent ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input.strip():
                continue

            print("\nTRACE_ATLAS: ", end="", flush=True)

            for event in agent(user_input):
                content = event.data.get("article") or event.data.get("content")
                if content:
                    print(f"\n[{event.type}]: {content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
