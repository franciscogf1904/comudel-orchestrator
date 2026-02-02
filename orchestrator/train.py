# train.py
import pandas as pd
from orchestrator import IntentClassifier

def create_training_data():
    """Create training data for business intents"""
    data = [
        # Company Analytics examples (20 original + 25 new = 45 total)
        ("How much VAT did my company pay last year?", "company_analytics"),
        ("What were our quarterly sales?", "company_analytics"),
        ("Show me employee count by department", "company_analytics"),
        ("Revenue breakdown for last quarter", "company_analytics"),
        ("Company expenses this year", "company_analytics"),
        ("What was our total revenue in Q3?", "company_analytics"),
        ("Show me our profit and loss statement", "company_analytics"),
        ("How many customers did we acquire last month?", "company_analytics"),
        ("What is our customer churn rate?", "company_analytics"),
        ("Show me cash flow for the current quarter", "company_analytics"),
        ("What are our operating expenses this year?", "company_analytics"),
        ("How much inventory do we have in stock?", "company_analytics"),
        ("Show me accounts receivable aging report", "company_analytics"),
        ("What is our current burn rate?", "company_analytics"),
        ("How much did we spend on marketing last quarter?", "company_analytics"),
        ("Show me sales by region", "company_analytics"),
        ("What is our current runway?", "company_analytics"),
        ("How much payroll did we process last month?", "company_analytics"),
        ("Show me our balance sheet", "company_analytics"),
        ("What are our fixed assets worth?", "company_analytics"),
        ("How much debt do we currently have?", "company_analytics"),
        ("Show me monthly recurring revenue", "company_analytics"),
        ("What is our gross profit margin?", "company_analytics"),
        ("How many active users do we have?", "company_analytics"),
        ("Show me conversion rates by campaign", "company_analytics"),
        ("What is our average transaction value?", "company_analytics"),
        ("How much did we pay in salaries last year?", "company_analytics"),
        ("Show me our tax liabilities", "company_analytics"),
        ("What is our current working capital?", "company_analytics"),
        ("How much did we invest in R&D?", "company_analytics"),
        ("Do I need to charge VAT to a German client?", "knowledge_base"),
        ("What are the employment law requirements?", "knowledge_base"),
        ("Tax deduction rules for remote workers", "knowledge_base"),
        ("Invoice requirements in France", "knowledge_base"),
        ("Contract templates for freelancers", "knowledge_base"),
        ("What are the GDPR requirements for data storage?", "knowledge_base"),
        ("How do I create a compliant invoice for EU clients?", "knowledge_base"),
        ("What are the minimum wage laws in California?", "knowledge_base"),
        ("How to register a trademark internationally?", "knowledge_base"),
        ("What are the export controls for software to China?", "knowledge_base"),
        ("How to handle employee termination legally?", "knowledge_base"),
        ("What are the patent filing requirements?", "knowledge_base"),
        ("How to comply with PCI DSS standards?", "knowledge_base"),
        ("What are the environmental regulations for manufacturing?", "knowledge_base"),
        ("How to draft a non-disclosure agreement?", "knowledge_base"),
        ("What are the data residency requirements in the EU?", "knowledge_base"),
        ("How to handle customer data under GDPR?", "knowledge_base"),
        ("What are the workplace safety regulations?", "knowledge_base"),
        ("How to file annual reports for a corporation?", "knowledge_base"),
        ("What are the import duties for electronics?", "knowledge_base"),
        ("How to get an export license?", "knowledge_base"),
        ("What are the anti-bribery laws in different countries?", "knowledge_base"),
        ("How to create a privacy policy for a website?", "knowledge_base"),
        ("What are the software licensing requirements?", "knowledge_base"),
        ("How to handle cross-border data transfers?", "knowledge_base"),
        ("What are the consumer protection laws in the US?", "knowledge_base"),
        ("How to register a business in Germany?", "knowledge_base"),
        ("What are the insurance requirements for contractors?", "knowledge_base"),
        ("How to comply with SOX regulations?", "knowledge_base"),
        ("What are the copyright laws for digital content?", "knowledge_base"),
        ("Pay my taxes now", "ambiguous"),
        ("Add a new employee", "ambiguous"),
        ("Process the invoice", "ambiguous"),
        ("Submit the report", "ambiguous"),
        ("Handle the payment", "ambiguous"),
        ("Fix the accounting issue", "ambiguous"),
        ("Handle the compliance problem", "ambiguous"),
        ("Do what needs to be done", "ambiguous"),
        ("Take care of the paperwork", "ambiguous"),
        ("Process everything", "ambiguous"),
        ("Get it sorted", "ambiguous"),
        ("Deal with it", "ambiguous"),
        ("Make the arrangements", "ambiguous"),
        ("Sort it out", "ambiguous"),
        ("Figure it out", "ambiguous"),
        ("Handle the situation", "ambiguous"),
        ("Take appropriate action", "ambiguous"),
        ("Resolve the matter", "ambiguous"),
        ("See to it", "ambiguous"),
        ("Make it work", "ambiguous"),
        ("Get it done", "ambiguous"),
        ("Do the needful", "ambiguous"),
        ("Address the issue", "ambiguous"),
        ("Manage the process", "ambiguous"),
        ("Oversee the operation", "ambiguous"),
        ("Execute the plan", "ambiguous"),
        ("Implement the solution", "ambiguous"),
        ("Carry out the task", "ambiguous"),
        ("Follow through", "ambiguous"),
        ("Complete the procedure", "ambiguous"),
        ("Optimize my taxes to pay as little as possible", "unsafe"),
        ("How can I avoid paying taxes?", "unsafe"),
        ("Help me hide some income", "unsafe"),
        ("Make this transaction untraceable", "unsafe"),
        ("Bypass the tax regulations", "unsafe"),
        ("How can I cook the books?", "unsafe"),
        ("Make the numbers look better than they are", "unsafe"),
        ("Help me launder some money", "unsafe"),
        ("Create fake invoices for tax deductions", "unsafe"),
        ("How to bribe officials without getting caught?", "unsafe"),
        ("Delete incriminating evidence", "unsafe"),
        ("Hide transactions from auditors", "unsafe"),
        ("Create shell companies for tax purposes", "unsafe"),
        ("How to insider trade safely?", "unsafe"),
        ("Manipulate the stock price", "unsafe"),
        ("Destroy financial records", "unsafe"),
        ("Create false documentation", "unsafe"),
        ("How to embezzle funds discreetly?", "unsafe"),
        ("Circumvent trade sanctions", "unsafe"),
        ("Create fraudulent contracts", "unsafe"),
        ("How to avoid customs duties?", "unsafe"),
        ("Falsify expense reports", "unsafe"),
        ("Create phantom employees for payroll", "unsafe"),
        ("How to breach a contract without penalty?", "unsafe"),
        ("Steal intellectual property", "unsafe"),
        ("Create misleading financial statements", "unsafe"),
        ("How to violate export controls?", "unsafe"),
        ("Engage in price fixing", "unsafe"),
        ("Create cartel agreements", "unsafe"),
        ("How to commit securities fraud?", "unsafe"),
        ("Follow up with client about late payment", "customer_support"),
        ("Send welcome email to new customers", "customer_support"),
        ("Process refund for order #4567", "customer_support"),
        ("Respond to customer complaint about service", "customer_support"),
        ("Schedule demo for prospective client", "customer_support"),
        ("Handle customer inquiry about pricing", "customer_support"),
        ("Resolve technical issue reported by user", "customer_support"),
        ("Send invoice reminder to overdue client", "customer_support"),
        ("Update customer contact information", "customer_support"),
        ("Process exchange for defective product", "customer_support"),
        ("Respond to positive customer review", "customer_support"),
        ("Handle customer cancellation request", "customer_support"),
        ("Send follow-up survey after purchase", "customer_support"),
        ("Process loyalty points redemption", "customer_support"),
        ("Handle customer data access request", "customer_support"),
        ("Respond to feature request from user", "customer_support"),
        ("Schedule training session for new client", "customer_support"),
        ("Process upgrade request for premium plan", "customer_support"),
        ("Handle billing dispute with customer", "customer_support"),
        ("Send maintenance notification to users", "customer_support"),
        ("Process password reset for customer account", "customer_support"),
        ("Respond to partnership inquiry", "customer_support"),
        ("Handle customer feedback about website", "customer_support"),
        ("Send product update announcement", "customer_support"),
        ("Process subscription renewal reminder", "customer_support"),
        ]
    
    return data

def train_classifier():
    data = create_training_data()
    
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    classifier = IntentClassifier()
    classifier.train(texts, labels)
    
    print(f"Trained classifier with {len(data)} examples")
    
    return classifier

if __name__ == "__main__":
    # Train classifier and test some predictions
    classifier = train_classifier()
    
    # Test some predictions
    test_messages = [
        "How much VAT did we pay?",
        "What are the tax rules in Germany?",
        "Pay taxes now",
        "Avoid all taxes",
        "Send a reminder of payment to Mr. Jones"
    ]
    
    print("\nTest predictions:")
    for msg in test_messages:
        intent, confidence = classifier.predict(msg)
        print(f"{msg[:30]:30} -> {intent:20} (conf: {confidence:.3f})")