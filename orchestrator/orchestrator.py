# orchestrator.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
from nltk.stem import PorterStemmer

INTENTS = [
    "company_analytics",
    "knowledge_base", 
    "ambiguous",
    "unsafe",
    "customer_support"
]

@dataclass
class SafetyAssessment:
    status: str  # "allowed", "flagged", "blocked"
    reason: Optional[str] = None
    confidence: float = 0.0

@dataclass
class ToolCall:
    tool: str
    action: str
    args: Dict[str, Any]
    
@dataclass
class PlanStep:
    step: str
    tool_call: Optional[ToolCall] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class DecisionObject:
    intent: str
    confidence: float
    needs_clarification: bool
    safety: SafetyAssessment
    plan: List[PlanStep]
    tool_results: List[ToolResult]
    final_response: str
    
    def to_dict(self):
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "safety": {
                "status": self.safety.status,
                "reason": self.safety.reason,
                "confidence": self.safety.confidence
            },
            "plan": [
                {
                    "step": step.step,
                    "tool": step.tool_call.tool if step.tool_call else None,
                    "args": step.tool_call.args if step.tool_call else {},
                    "needs_clarification": step.needs_clarification,
                    "clarification_question": step.clarification_question
                }
                for step in self.plan
            ],
            "tool_results": [
                {
                    "tool": result.tool,
                    "ok": result.ok,
                    "data": result.data,
                    "error": result.error
                }
                for result in self.tool_results
            ],
            "final_response": self.final_response
        }

class IntentClassifier:
    def __init__(self, ambiguous_word_threshold=5, ambiguity_boost=0.10):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.encoder = LabelEncoder()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.stemmer = PorterStemmer()
        self._trained = False
        self.ambiguous_word_threshold = ambiguous_word_threshold
        self.ambiguity_boost = ambiguity_boost
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Lowercase
        text = text.lower()
        
        # Stemming
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(stemmed_words)
    
    def train(self, texts, labels):
        # Preprocess all texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit and transform with TF-IDF
        X = self.vectorizer.fit_transform(preprocessed_texts)
        
        # Encode labels
        y = self.encoder.fit_transform(labels)
        
        # Train model
        self.model.fit(X, y)
        self._trained = True
        
        return self
    
    def predict(self, text):
        if not self._trained:
            raise ValueError("Classifier must be trained first")
        
        # Preprocess
        preprocessed = self.preprocess_text(text)
        
        # Transform
        X = self.vectorizer.transform([preprocessed])
        
        # Predict
        proba = self.model.predict_proba(X)[0]
        
        
        word_count = len(preprocessed.split())
        
        # Apply ambiguity bias if message is short
        if word_count <= self.ambiguous_word_threshold:
            # Find ambiguous intent index
            try:
                ambiguous_idx = list(self.encoder.classes_).index("ambiguous")
                
                # Calculate boost: fewer words = more boost
                missing_words = self.ambiguous_word_threshold - word_count
                boost = self.ambiguity_boost * missing_words
                
                # Apply boost to ambiguous probability
                proba[ambiguous_idx] += boost
                
                # Reduce other probabilities proportionally
                non_ambiguous_mask = np.ones(len(proba), dtype=bool)
                non_ambiguous_mask[ambiguous_idx] = False
                total_non_ambiguous = proba[non_ambiguous_mask].sum()
                
                if total_non_ambiguous > 0:
                    reduction_factor = 1 - (boost / total_non_ambiguous)
                    proba[non_ambiguous_mask] *= reduction_factor
                
                # Ensure probabilities sum to 1
                proba = proba / proba.sum()
                
            except ValueError:
                # "ambiguous" not in classes
                pass
        
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        intent = self.encoder.inverse_transform([pred_idx])[0]
        
        return intent, confidence

class SafetyChecker:
    def __init__(self):
        self.unsafe_patterns = [
            r"optimize.*tax.*as little as possible",
            r"avoid.*tax",
            r"evade.*tax",
            r"evasion",
            r"pay.*no tax",
            r"illegal.*tax",
            r"cheat.*tax",
            r"hide.*income",
            r"underreport.*income",
            r"get away"
        ]
        
        self.sensitive_patterns = [
            r"password",
            r"credit.*card",
            r"ssn|social.*security",
            r"bank.*account",
            r"confidential",
            r"secret"
        ]
        
    def assess(self, text, intent):
        text_lower = text.lower()
        
        # Check for unsafe tax evasion patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text_lower):
                return SafetyAssessment(
                    status="blocked",
                    reason="Potential tax evasion detected",
                    confidence=0.85
                )
        
        # Check for sensitive information requests
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text_lower):
                return SafetyAssessment(
                    status="flagged",
                    reason="Sensitive information requested",
                    confidence=0.75
                )
        
        # Ambiguous requests might need careful handling
        if intent == "ambiguous":
            return SafetyAssessment(
                status="flagged",
                reason="Request is ambiguous and needs clarification",
                confidence=0.60
            )
        
        return SafetyAssessment(
            status="allowed",
            reason=None,
            confidence=0.95
        )

class ToolExecutor:
    
    def execute(self, tool_call):
        if tool_call.tool == "company_data":
            return self._execute_company_data(tool_call)
        elif tool_call.tool == "knowledge_base":
            return self._execute_knowledge_base(tool_call)
        elif tool_call.tool == "customer_support_system":  # NEW
            return self._execute_customer_support(tool_call)
        else:
            return ToolResult(
                tool=tool_call.tool,
                ok=False,
                data={},
                error=f"Unknown tool: {tool_call.tool}"
            )
    
    def _execute_company_data(self, tool_call):
        action = tool_call.action
        
        if action == "query_vat":
            return ToolResult(
                tool="company_data",
                ok=True,
                data={
                    "vat_paid_last_year": "€15,250",
                    "period": "2023",
                    "currency": "EUR",
                    "breakdown": {
                        "Q1": "€3,800",
                        "Q2": "€3,900",
                        "Q3": "€3,750",
                        "Q4": "€3,800"
                    }
                }
            )
        elif action == "query_employee_count":
            return ToolResult(
                tool="company_data",
                ok=True,
                data={
                    "current_employees": 42,
                    "departments": {
                        "Engineering": 15,
                        "Sales": 10,
                        "Operations": 8,
                        "Finance": 5,
                        "HR": 4
                    }
                }
            )
        else:
            return ToolResult(
                tool="company_data",
                ok=True,
                data={
                    "message": f"Mock data for {action}",
                    "value": "Sample response"
                }
            )
    
    def _execute_knowledge_base(self, tool_call):
        action = tool_call.action
        
        if action == "query_vat_rules":
            return ToolResult(
                tool="knowledge_base",
                ok=True,
                data={
                    "country": "Germany",
                    "vat_rate": "19% standard rate",
                    "threshold": "€22,000 annual turnover",
                    "requirements": "Must register for VAT if threshold exceeded",
                    "source": "EU VAT Directive 2006/112/EC"
                }
            )
        elif action == "query_employment_law":
            return ToolResult(
                tool="knowledge_base",
                ok=True,
                data={
                    "topic": "New employee registration",
                    "requirements": [
                        "Employment contract",
                        "Tax registration",
                        "Social security registration",
                        "Work permit (if applicable)"
                    ],
                    "timeline": "Before first working day",
                    "jurisdiction": "EU/National regulations"
                }
            )
        else:
            return ToolResult(
                tool="knowledge_base",
                ok=True,
                data={
                    "answer": f"Mock knowledge base response for {action}",
                    "sources": ["Internal KB v1.0", "Compliance Database"]
                }
            )
    def _execute_customer_support(self, tool_call):
        action = tool_call.action
        
        if action == "create_ticket":
            return ToolResult(
                tool="customer_support_system",
                ok=True,
                data={
                    "ticket_id": "CS-" + str(hash(message))[:6],
                    "status": "open",
                    "priority": tool_call.args.get("priority", "medium"),
                    "assigned_to": "Support Agent",
                    "estimated_resolution": "24-48 hours"
                }
            )
        elif action == "schedule_followup":
            return ToolResult(
                tool="customer_support_system",
                ok=True,
                data={
                    "followup_id": "FU-" + str(hash(message))[:6],
                    "scheduled_time": "Tomorrow 10:00 AM",
                    "channel": tool_call.args.get("channel", "email"),
                    "recipient": "Customer",
                    "task": tool_call.args.get("task", "follow_up")
                }
            )
        elif action == "process_refund":
            return ToolResult(
                tool="customer_support_system",
                ok=True,
                data={
                    "refund_id": "RF-" + str(hash(message))[:6],
                    "status": "pending_approval",
                    "amount": "To be determined",
                    "reason": tool_call.args.get("description", "Customer request"),
                    "estimated_processing": "3-5 business days"
                }
            )
        else:
            return ToolResult(
                tool="customer_support_system",
                ok=True,
                data={
                    "support_case_id": "SC-" + str(hash(message))[:8],
                    "status": "in_progress",
                    "action_taken": "Request logged in support system",
                    "next_steps": "Customer will be contacted within 24 hours"
                }
            )

class ResponseComposer:    
    def compose(self, intent, tool_results, needs_clarification, safety):
        if safety.status == "blocked":
            return self._compose_blocked_response(safety)
        
        if needs_clarification:
            return self._compose_clarification_response(intent)
        
        if intent == "company_analytics":
            return self._compose_company_response(tool_results)
        elif intent == "knowledge_base":
            return self._compose_knowledge_response(tool_results)
        elif intent == "ambiguous":
            return self._compose_ambiguous_response()
        elif intent == "customer_support":
            return self._compose_customer_support_response(tool_results)
        else:
            return self._compose_general_response(tool_results)
    
    def _compose_blocked_response(self, safety):
        return f"I cannot assist with this request. Reason: {safety.reason}. If you have legitimate tax planning questions, please consult with a certified tax advisor."
    
    def _compose_clarification_response(self, intent):
        if intent == "ambiguous":
            return "I need more information to help you. Please specify"
        return "I need some clarification to proceed with your request. Could you provide more details?"
    
    def _compose_company_response(self, tool_results):
        if not tool_results:
            return "I've analyzed your company data request. What specific metrics would you like to see?"
        
        result = tool_results[0]
        if result.tool == "company_data":
            if "vat_paid_last_year" in result.data:
                return f"Based on our records, your company paid {result.data['vat_paid_last_year']} in VAT last year. The quarterly breakdown is: Q1: {result.data['breakdown']['Q1']}, Q2: {result.data['breakdown']['Q2']}, Q3: {result.data['breakdown']['Q3']}, Q4: {result.data['breakdown']['Q4']}."
        
        return "Here are the company analytics you requested."
    
    def _compose_knowledge_response(self, tool_results):
        if not tool_results:
            return "I've retrieved information from our knowledge base. What specific question can I help with?"
        
        result = tool_results[0]
        if result.tool == "knowledge_base":
            if "vat_rate" in result.data:
                return f"For Germany: The standard VAT rate is {result.data['vat_rate']}. Registration is required if annual turnover exceeds {result.data['threshold']}."
        
        return "Here's the information from our knowledge base."
    def _compose_customer_support_response(self, tool_results):
        if not tool_results:
            return "I've logged your customer support request. How else can I assist with customer-related matters?"
        
        result = tool_results[0]
        if result.tool == "customer_support_system":
            if "ticket_id" in result.data:
                return f"Customer support ticket created successfully. Ticket ID: {result.data['ticket_id']}. Priority: {result.data['priority']}. Estimated resolution: {result.data['estimated_resolution']}."
            elif "refund_id" in result.data:
                return f"Refund request processed. Refund ID: {result.data['refund_id']}. Status: {result.data['status']}. Estimated processing time: {result.data['estimated_processing']}."
            elif "followup_id" in result.data:
                return f"Follow-up scheduled. Follow-up ID: {result.data['followup_id']}. Scheduled for: {result.data['scheduled_time']} via {result.data['channel']}."
        
        return "Your customer support request has been logged. A support agent will contact you shortly."
    
    def _compose_ambiguous_response(self):
        return "Your request needs clarification. Could you provide more specific details about what you'd like to accomplish?"
    
    def _compose_general_response(self, tool_results):
        return "I've processed your request. Here are the results."

class Orchestrator:
    
    def __init__(self):
        self.classifier = IntentClassifier()
        self.safety_checker = SafetyChecker()
        self.tool_executor = ToolExecutor()
        self.response_composer = ResponseComposer()
        
        # Train with initial examples
        self._train_classifier()
    
    def _train_classifier(self):
        from train import create_training_data
        
        data = create_training_data()
        texts = [item[0] for item in data]
        labels = [item[1] for item in data]
        
        self.classifier.train(texts, labels)
    
    def process(self, user_message):
        
        intent, confidence = self.classifier.predict(user_message)
        safety = self.safety_checker.assess(user_message, intent)
        plan = self._create_plan(intent, user_message, safety)
        tool_results = []
        if safety.status == "allowed" and not plan[0].needs_clarification:
            for step in plan:
                if step.tool_call:
                    result = self.tool_executor.execute(step.tool_call)
                    tool_results.append(result)
        
        needs_clarification = any(step.needs_clarification for step in plan)
        
        final_response = self.response_composer.compose(
            intent, tool_results, needs_clarification, safety
        )
        
        decision = DecisionObject(
            intent=intent,
            confidence=float(confidence),
            needs_clarification=needs_clarification,
            safety=safety,
            plan=plan,
            tool_results=tool_results,
            final_response=final_response
        )
        
        return decision
    
    def _create_plan(self, intent, message, safety):
        plan = []
        
        if safety.status == "blocked":
            plan.append(PlanStep(
                step="safety_block",
                needs_clarification=False
            ))
            return plan
        
        if intent == "company_analytics":
            if "vat" in message.lower() and "pay" in message.lower():
                plan.append(PlanStep(
                    step="query_company_vat_data",
                    tool_call=ToolCall(
                        tool="company_data",
                        action="query_vat",
                        args={"metric": "vat_paid", "period": "last_year"}
                    )
                ))
            else:
                plan.append(PlanStep(
                    step="generic_company_query",
                    tool_call=ToolCall(
                        tool="company_data",
                        action="query_general",
                        args={"query": message}
                    )
                ))
                
        elif intent == "knowledge_base":
            if "german" in message.lower() or "germany" in message.lower():
                plan.append(PlanStep(
                    step="query_german_vat_rules",
                    tool_call=ToolCall(
                        tool="knowledge_base",
                        action="query_vat_rules",
                        args={"country": "Germany", "topic": "VAT"}
                    )
                ))
            elif "employee" in message.lower():
                plan.append(PlanStep(
                    step="query_employment_law",
                    tool_call=ToolCall(
                        tool="knowledge_base",
                        action="query_employment_law",
                        args={"topic": "new_employee"}
                    )
                ))
            else:
                plan.append(PlanStep(
                    step="generic_knowledge_query",
                    tool_call=ToolCall(
                        tool="knowledge_base",
                        action="query_general",
                        args={"query": message}
                    )
                ))
                
        elif intent == "ambiguous":
            plan.append(PlanStep(
                step="request_clarification",
                needs_clarification=True,
                clarification_question="Please provide more specific details about your request."
            ))
            
        elif intent == "unsafe":
            plan.append(PlanStep(
                step="safety_review",
                needs_clarification=False
            ))
        elif intent == "customer_support":
            if "complaint" in message.lower() or "issue" in message.lower():
                plan.append(PlanStep(
                    step="handle_customer_complaint",
                    tool_call=ToolCall(
                        tool="customer_support_system",
                        action="create_ticket",
                        args={"type": "complaint", "priority": "high", "description": message}
                    )
                ))
            elif "follow up" in message.lower() or "reminder" in message.lower():
                plan.append(PlanStep(
                    step="schedule_follow_up",
                    tool_call=ToolCall(
                        tool="customer_support_system",
                        action="schedule_followup",
                        args={"task": "follow_up", "description": message, "channel": "email"}
                    )
                ))
            elif "refund" in message.lower() or "return" in message.lower():
                plan.append(PlanStep(
                    step="process_refund_request",
                    tool_call=ToolCall(
                        tool="customer_support_system",
                        action="process_refund",
                        args={"request_type": "refund", "description": message}
                    )
                ))
            else:
                plan.append(PlanStep(
                    step="generic_customer_support",
                    tool_call=ToolCall(
                        tool="customer_support_system",
                        action="handle_request",
                        args={"request": message, "category": "general"}
                    )
                ))
            
        return plan
