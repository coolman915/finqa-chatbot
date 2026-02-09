"""KG triplet extraction prompt (Paper 2: Structure First, Reason Next)."""

KG_EXTRACTION_SYSTEM = """You are a financial knowledge graph extraction system.

Given financial text and/or tables, extract structured knowledge graph triplets using this schema:

Subject: "METRIC" or "METRIC:Company"
  Examples: "Revenue", "Revenue:AppleInc", "NetIncome:MicrosoftCorp"

Relation: describes when/how the value applies
  - HAS_VALUE_IN_YYYY (e.g., HAS_VALUE_IN_2017)
  - HAS_CHANGE_FROM_YYYY_TO_YYYY
  - HAS_PERCENTAGE_IN_YYYY
  - IS_COMPONENT_OF

Object: "value unit"
  Examples: "5829 million USD", "27.3 percent", "yes"

Output a JSON array of triplets. Each triplet has:
{
  "subject": "...",
  "relation": "...",
  "object": "...",
  "entity_type": "financial_metric" | "company" | "ratio" | "count",
  "company": "..." (if identifiable),
  "period": "YYYY" (extracted from relation),
  "value": <float>,
  "unit": "..."
}

Focus on numeric facts that are directly relevant to the question.
Output ONLY the JSON array, no other text."""

KG_EXTRACTION_FEW_SHOT = [
    {
        "context": "Table:\n| ( in millions ) | 2017 | 2016 |\n| operating income | 11503 | 10815 |\n| net revenue | 7000 | 6300 |",
        "question": "what was the change in operating income from 2016 to 2017?",
        "triplets": [
            {
                "subject": "OperatingIncome",
                "relation": "HAS_VALUE_IN_2017",
                "object": "11503 million",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2017",
                "value": 11503.0,
                "unit": "million"
            },
            {
                "subject": "OperatingIncome",
                "relation": "HAS_VALUE_IN_2016",
                "object": "10815 million",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2016",
                "value": 10815.0,
                "unit": "million"
            },
        ],
    },
    {
        "context": "Text: total net revenue increased $ 693 million , or 11% , to $ 7.0 billion .\nTable:\n| year | net revenue |\n| 2006 | 7.0 |\n| 2005 | 6.3 |",
        "question": "what is the percent change in total net revenue from 2005 to 2006?",
        "triplets": [
            {
                "subject": "NetRevenue",
                "relation": "HAS_VALUE_IN_2006",
                "object": "7.0 billion",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2006",
                "value": 7.0,
                "unit": "billion"
            },
            {
                "subject": "NetRevenue",
                "relation": "HAS_VALUE_IN_2005",
                "object": "6.3 billion",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2005",
                "value": 6.3,
                "unit": "billion"
            },
        ],
    },
]
