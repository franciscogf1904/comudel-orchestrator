# comudel-orchestrator
Comudel recruitment orchestrator challenge
# Project Structure
```
Orchestrator------------->run.py
                |
                |
                ├-------->orchestrator.py
                |
                |
                |-------->train.py
```
# Architecture

User Message -> Intent Classifier -> Safety Checker -> Planner -> Tool Executor -> Responde Composer -> Final Response and JSON Object

# Component Responsibilities

## Intent Classifier - As the name says, this component classifies the user message according to five intents:

Company Analytics, 

Knowledge Base,

Customer Support, 

Ambiguous,

Unsafe. 

While the first three classifications depend only on the user's wish, the last two are applied to requests where the system can't help, either because it doesn't have the means to do so or because it has blocked the user request. Before the system classifies a request, it preprocesses it through 4 different phases:

Ponctuation removal, capitalization uniforming and word stemming

TF-IDF Vectorization,

Word count (the lesser words the request has, the higher the chance it does not have all the necessary information and therefore can be more ambiguous.

## Safety Checker - Detects unsafe or sensitive requests by looking for specific (combinations of) words that might indicate illegal requests or sensitive information

## Planner - Creates an execution plan with a specific tool

## Tool Execution - Show the result of applying said tool

## Response Composer - Creates both the JSON Object and the final response for the user

# Tradeoffs

While using a combination of a classifier with examples and a pattern checker can catch specific words and phrases of common requests, sophisticated and unique requests may elude the classifier. However, given the lack of real request examples to train a useful ML classifier, this was the safest approach.

Adding the word count bias helps with smaller messages that dont have a lot of information, but is not helpful with requests that pass the five word threshold but still do not contain all the relevant information

# Limitations

The lack of examples from clients results in less than ideal training. However, since our approach is easily scalable and can even be changed to receive csv or txt files as inputs with human-classified requests for training, this would not be a problem on a larger scale project.

TF-IDF does not take semantic understanding into account, so synonyms of words would not be classified the same. THis could be solved by incorporating an LLM into our classifier.

A big part of the JSON Object is mocked and both the Knowledge Base and Company Analytics have fake information usable only for demonstration purposes. To be usable in a real word scenario, the classifier needs to be incorporated,

# Future Work (1 Week)

Create more training examples, ideally by gathering client requests to get the most realistic training data

Add semantic understanding, to similarize identical requests more easily

If possible, look into each client so that we know what tools and plans are available for each request





