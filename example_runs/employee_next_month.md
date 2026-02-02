# Input

"Add a new employee starting next month"

# JSON
```
{
  "intent": "ambiguous",
  "confidence": 0.3147199612708313,
  "needs_clarification": true,
  "safety": {
    "status": "flagged",
    "reason": "Request is ambiguous and needs clarification",
    "confidence": 0.6
  },
  "plan": [
    {
      "step": "request_clarification",
      "tool": null,
      "args": {},
      "needs_clarification": true,
      "clarification_question": "Please provide more specific details about your request."
    }
  ],
  "tool_results": [],
  "final_response": "I need more information to help you. Please specify"
}
```
