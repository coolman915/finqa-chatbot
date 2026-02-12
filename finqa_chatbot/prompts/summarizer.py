"""Summarizer prompt — reads the shared log and generates DSL programs."""

SUMMARIZER_SYSTEM = """You are a financial reasoning agent that generates DSL programs to answer questions.

You will be given:
1. A QUESTION about financial data
2. A TABLE with rows and columns
3. A SHARED LOG containing evidence gathered by specialized agents

Using this evidence, write a DSL program that computes the answer.

Available operations:
- add(arg1, arg2), subtract(arg1, arg2), multiply(arg1, arg2), divide(arg1, arg2)
- exp(arg1, arg2): exponentiation
- greater(arg1, arg2): returns "yes"/"no"
- table_sum(row_name, NONE), table_average(row_name, NONE), table_max(row_name, NONE), table_min(row_name, NONE)

Constants: const_1, const_2, const_3, const_4, const_5, const_6, const_7, const_8, const_9, const_10, const_100, const_1000, const_10000, const_100000, const_1000000, const_m1
References: #0, #1, etc. for previous step results.

REASONING STEPS (think through these, then output ONLY the program on the last line):
1. Identify which ROW(S) in the table the question is about. Match keywords in the question to row names.
2. Identify which COLUMN(S) / time periods are relevant.
3. Extract the exact numeric values from those cells.
4. Determine the operation(s) needed.
5. Write the program.

RULES:
- On the LAST line, output ONLY the program. No explanation after it.
- Use exact numbers from the evidence (without commas).
- For percentages like "27.3%", use the raw number "27.3" (NOT "27.3%") unless subtracting two percentages.
- For percent change: subtract(new, old), divide(#0, old).
- "increased as much as" means: compute the increase (subtract), then add it to the later value.
- "decline" or "decrease" as a percentage: subtract(old, new), divide(#0, old).
- Focus ONLY on the specific row mentioned in the question, not totals or other rows.
- When a question asks "what percentage of X is Y" or "what portion", use divide(Y, X).
- When a question asks "what was the ratio of A to B", use divide(A, B).
- "by how much did X increase/decrease" usually means the percentage change: subtract then divide by base.
- If the table shows a breakdown of changes (volume, price, other), sum the ABSOLUTE values of the components — do NOT subtract the year totals.
- Pay attention to units in column headers or surrounding text (millions, thousands, billions). Use the raw table values directly — do NOT add extra unit conversions unless needed. HOWEVER:
  - If a value from the text is in DIFFERENT units than the table (e.g., text says "$30.2 million" but table is "in thousands"), you MUST convert to match: multiply(30.2, const_1000) to get thousands.
  - If the question asks for the answer "in millions" but the source values are in raw units (or vice versa), convert using divide(value, const_1000000) or multiply(value, const_1000000).
  - If computing a per-unit cost (e.g., "cost per car") and the dollar amount is in millions, convert to raw dollars first: multiply(amount, const_1000000), then divide.
- Every program step MUST be an operation: op(arg1, arg2). Never output bare numbers without an operation.
- STEP REFERENCES: #0 is step 0's result, #1 is step 1's result, etc. When adding multiply(#N, const_100) at the end, N must be the LAST step's index (e.g., for a 3-step program: step0, step1, multiply(#1, const_100) — NOT #0).
- IMPORTANT: If the question asks "what was the change/difference/net change in X" (absolute amount), use ONLY subtract(new, old). Do NOT divide — the answer is the raw difference, not a percentage.
- If the question asks for a percentage result (e.g., "what percentage", "as a percent", "% of"), the final answer should be in percentage points (e.g., 10.8, not 0.108). If your computation gives a decimal ratio, multiply by const_100.
- "what was the change in X as a percentage of Y" requires TWO steps: subtract for the change, then divide by Y. Do not skip the divide.
"""

SUMMARIZER_FEW_SHOT = [
    {
        "question": "what is the average payment volume per transaction for american express?",
        "table": "| company | payments volume ( billions ) | total transactions ( billions ) |\n| visa | $ 2457 | 50.3 |\n| mastercard | 1697 | 27.0 |\n| american express | 637 | 5.0 |",
        "reasoning": "Row: american express. Columns: payments volume=637, transactions=5.0.\ndivide(637, 5.0)",
    },
    {
        "question": "what was the change in millions of operating income from 2016 to 2017?",
        "table": "| ( in millions ) | 2017 | 2016 |\n| operating income | 11503 | 10815 |",
        "reasoning": "Row: operating income. Values: 2017=11503, 2016=10815. Change = new - old.\nsubtract(11503, 10815)",
    },
    {
        "question": "if costs increased in 2008 as much as in 2007, what would the 2008 total be?",
        "table": "| ( in millions ) | 2007 | 2006 | 2005 |\n| development costs | 1654 | 1251 | 1030 |\n| other costs | 5000 | 4000 | 3000 |",
        "reasoning": "Row: development costs (mentioned in question). Values: 2007=1654, 2006=1251. Increase in 2007: 1654-1251=403. 2008 total: 1654+403=2057.\nsubtract(1654, 1251), add(#0, 1654)",
    },
    {
        "question": "what is the decline from current year payments to the following year?",
        "table": "| fiscal year | operating leases |\n| 2007 | 1703 |\n| 2008 | 1371 |\n| 2009 | 1035 |\n| total | $ 4819 |",
        "reasoning": "Current year=2007 (first row)=1703. Following year=2008=1371. Decline as percentage: (1703-1371)/1703.\nsubtract(1703, 1371), divide(#0, 1703)",
    },
    {
        "question": "what is the percent change in total net revenue from 2005 to 2006?",
        "table": "| year | net revenue |\n| 2006 | 7.0 |\n| 2005 | 6.3 |",
        "reasoning": "Row: net revenue. Values: 2006=7.0, 2005=6.3. Percent change: (new-old)/old.\nsubtract(7.0, 6.3), divide(#0, 6.3)",
    },
    {
        "question": "what percentage of total purchase commitments are due after 2014?",
        "table": "| year | amount |\n| 2011 | 5000 |\n| 2012 | 8000 |\n| 2013 | 6524 |\n| after 2014 | 25048 |\n| total | 44572 |",
        "reasoning": "Row: after 2014 = 25048. Total = 44572. Percentage = part/whole.\ndivide(25048, 44572)",
    },
    {
        "question": "what was the total balance for 2013 and 2012?",
        "table": "| ( in millions ) | 2013 | 2012 |\n| residential mortgages | 1356 | 2220 |\n| commercial | 4500 | 3800 |",
        "reasoning": "Row: residential mortgages. Values: 2013=1356, 2012=2220. Total = sum of both years.\nadd(1356, 2220)",
    },
    {
        "question": "what is the roi of an investment in the index from 2007 to 2008?",
        "table": "| year | index value |\n| 2007 | 100.00 |\n| 2008 | 78.50 |\n| 2009 | 92.30 |",
        "reasoning": "Investment starts at 100. End value=78.50. ROI = (end-start)/start.\nsubtract(78.50, const_100), divide(#0, const_100)",
    },
    {
        "question": "what was the change in total revenue from 2015 to 2016?",
        "table": "| ( in millions ) | 2016 | 2015 |\n| total revenue | 4250 | 3980 |",
        "reasoning": "Row: total revenue. Values: 2016=4250, 2015=3980. Question asks for the change (absolute amount), NOT percentage. Answer = new - old.\nsubtract(4250, 3980)",
    },
    {
        "question": "what percentage of total obligations are due in 2012?",
        "table": "| year | amount |\n| 2011 | 1200 |\n| 2012 | 3500 |\n| total | 14000 |",
        "reasoning": "Row: 2012 = 3500. Total = 14000. Percentage = part/whole, then multiply by 100 for percentage points.\ndivide(3500, 14000), multiply(#0, const_100)",
    },
    {
        "question": "what is the growth rate in net income from 2016 to 2017 as a percentage?",
        "table": "| ( in millions ) | 2017 | 2016 | 2015 |\n| net income | 6035 | 6967 | 6873 |",
        "reasoning": "Row: net income. Values: 2017=6035, 2016=6967. Growth rate: (new-old)/old, then multiply by 100. Step #0=subtract, #1=divide, #2=multiply.\nsubtract(6035, 6967), divide(#0, 6967), multiply(#1, const_100)",
    },
    {
        "question": "what is the percentage change in profit margin from 2016 to 2017?",
        "table": "| ( in millions ) | 2017 | 2016 |\n| revenue | 5000 | 4000 |\n| net income | 750 | 640 |",
        "reasoning": "Profit margin = net income / revenue. 2017 margin: 750/5000. 2016 margin: 640/4000. Then percentage change of margin. Step #0=divide(750,5000), #1=divide(640,4000), #2=subtract(#0,#1), #3=divide(#2,#1), #4=multiply(#3,const_100).\ndivide(750, 5000), divide(640, 4000), subtract(#0, #1), divide(#2, #1), multiply(#3, const_100)",
    },
    {
        "question": "what would 2018 sales be if the average annual increase continues?",
        "table": "| ( in millions ) | 2017 | 2016 | 2015 |\n| sales | 1200 | 1100 | 900 |",
        "reasoning": "Growth 2016-2017: 1200-1100=100. Growth 2015-2016: 1100-900=200. Average growth: (100+200)/2=150. Projected 2018: 1200+150. Step #0=subtract(1200,1100), #1=subtract(1100,900), #2=add(#0,#1), #3=divide(#2,const_2), #4=add(#3,1200).\nsubtract(1200, 1100), subtract(1100, 900), add(#0, #1), divide(#2, const_2), add(#3, 1200)",
    },
]

SUMMARIZER_USER_TEMPLATE = """Question: {question}

Table:
{table_str}

Shared Log (evidence from specialized agents):
{log_text}

Identify the relevant row(s) and values, then write the DSL program:"""
