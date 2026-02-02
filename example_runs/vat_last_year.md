# Input 

"How much VAT did my company pay last year?"

# JSON
```
{
  "intent": "company_analytics",
  "confidence": 0.92,
  "needs_clarification": false,
  "safety": {
    "status": "allowed",
    "reason": null,
    "confidence": 0.95
  },
  "plan": [
    {
      "step": "query_company_vat_data",
      "tool": "company_data",
      "args": {
        "metric": "vat_paid",
        "period": "last_year"
      },
      "needs_clarification": false,
      "clarification_question": null
    }
  ],
  "tool_results": [
    {
      "tool": "company_data",
      "ok": true,
      "data": {
        "vat_paid_last_year": "€15,250",
        "period": "2023",
        "currency": "EUR",
        "breakdown": {
          "Q1": "€3,800",
          "Q2": "€3,900",
          "Q3": "€3,750",
          "Q4": "€3,800"
        }
      },
      "error": null
    }
  ],
  "final_response": "Based on our records, your company paid €15,250 in VAT last year. The quarterly breakdown is: Q1: €3,800, Q2: €3,900, Q3: €3,750, Q4: €3,800."
}
```
