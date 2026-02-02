# run.py
#!/usr/bin/env python3
import sys
import json
from orchestrator import Orchestrator

def main():
    if len(sys.argv) < 2:
        print("Invalid command. Usage: python run.py \"your message here\"")
        sys.exit(1)
    
    user_message = sys.argv[1]
    
    orchestrator = Orchestrator()
    decision = orchestrator.process(user_message)
    
    # Print decision JSON
    print("\n=== DECISION OBJECT ===")
    print(json.dumps(decision.to_dict(), indent=2))
    
    # Print final response
    print("\n=== FINAL RESPONSE ===")
    print(decision.final_response)

if __name__ == "__main__":
    main()