"""
All prompts and constants for the "Can India Become Rich Before It Grows Old?" project.
This file consolidates prompts previously spread across multiple files.
"""

from datetime import datetime

# Core economic constants
HIGH_INCOME_THRESHOLD_NOMINAL = 13000  # GNI per capita (nominal) threshold to be considered "rich" in USD
HIGH_INCOME_THRESHOLD_PPP = 13000  # GNI per capita (PPP) threshold to be considered "rich" in USD
DEMOGRAPHIC_PEAK_YEAR = 2035  # When India's working-age population share begins to decline

# Current economic values (will be updated by verifier)
CURRENT_GNI_NOMINAL = 2480.8  # GNI per capita (nominal) in USD
CURRENT_GNI_PPP = 7340  # GNI per capita (PPP) in USD

# Verifier settings
VERIFIER_INTERVAL_DAYS = 30  # Run verifier every 30 days

# API configuration
API_URL = "https://api.perplexity.ai/chat/completions"
API_MODEL = "sonar"
API_TEMPERATURE = 0.2
API_TOP_P = 0.9

# Data sources
PRIMARY_SOURCES = [
    "World Bank: https://data.worldbank.org",
    "International Monetary Fund (IMF): https://www.imf.org/en/Countries/IND",
    "United Nations – Population Division: https://population.un.org/wpp",
    "Ministry of Finance, Government of India: https://www.indiabudget.gov.in",
    "NITI Aayog: https://www.niti.gov.in",
    "Reserve Bank of India (RBI): https://rbi.org.in",
    "Ministry of Statistics and Programme Implementation (MoSPI): https://mospi.gov.in"
]

# For use in prompts
SOURCES_FORMATTED = "\n".join([f"- {source.split(':')[0]}" for source in PRIMARY_SOURCES])

# Helper function to calculate required growth rate
def calculate_growth_rate(current_value, target_value, years):
    """Calculate compound annual growth rate required to reach target from current value in given years"""
    return ((target_value / current_value) ** (1/years) - 1) * 100

# Calculate required growth rates
YEARS_TO_DEMOGRAPHIC_PEAK = DEMOGRAPHIC_PEAK_YEAR - datetime.now().year
REQUIRED_GROWTH_RATE_NOMINAL = calculate_growth_rate(CURRENT_GNI_NOMINAL, HIGH_INCOME_THRESHOLD_NOMINAL, YEARS_TO_DEMOGRAPHIC_PEAK)
REQUIRED_GROWTH_RATE_PPP = calculate_growth_rate(CURRENT_GNI_PPP, HIGH_INCOME_THRESHOLD_PPP, YEARS_TO_DEMOGRAPHIC_PEAK)

# Default economic facts (fallbacks if verification fails)
DEFAULT_FACTS = {
    "current_gni_per_capita_nominal": CURRENT_GNI_NOMINAL,
    "current_gni_per_capita_ppp": CURRENT_GNI_PPP,
    "required_growth_rate_nominal": REQUIRED_GROWTH_RATE_NOMINAL,
    "required_growth_rate_ppp": REQUIRED_GROWTH_RATE_PPP,
    "projected_growth_rate_nominal": 7.0,
    "projected_growth_rate_ppp": 5.0,
    "demographic_peak_year": DEMOGRAPHIC_PEAK_YEAR,
    "high_income_threshold_nominal": HIGH_INCOME_THRESHOLD_NOMINAL,
    "high_income_threshold_ppp": HIGH_INCOME_THRESHOLD_PPP,
    "years_to_demographic_peak": YEARS_TO_DEMOGRAPHIC_PEAK
}

#-----------------------------------------------------------------------------
# System Prompts
#-----------------------------------------------------------------------------

# System prompt for the verifier agent
VERIFIER_SYSTEM_PROMPT = """You are an expert economist specializing in India's economic and demographic development.
Your task is to verify key economic facts and figures related to India's growth trajectory.
Provide only verified, up-to-date data from authoritative sources.
Focus exclusively on facts and figures - no analysis or recommendations.
Format your responses in a concise, structured manner suitable for data processing."""

# System prompt for the data agent
DATA_SYSTEM_PROMPT = """You are an expert economic analyst specializing in India's growth and demographic transition.
Provide factual, data-driven analysis about India's prospects of becoming rich before it grows old.
Base insights strictly on current data, projections, and credible research.

Use only established sources:
{sources}

Do not speculate. Prioritize data-backed reasoning with specific figures and percentages.
Ensure all sections include citations to authoritative sources.
Format content formally - no conversational phrases, introductions, or filler text.
All content must be direct, precise, and objective in the style of authoritative economic publications.""".format(
    sources=SOURCES_FORMATTED
)

# System prompt for the review agent
REVIEW_SYSTEM_PROMPT = """You are a senior editor specializing in economic analysis and clear communication.
Review economic content for clarity, consistency, factual accuracy, and rigor.
Provide specific, actionable feedback on how to improve content quality without changing the underlying assessment.
Focus on improving precision, logical flow, and evidential support rather than stylistic preferences."""

#-----------------------------------------------------------------------------
# User Prompts
#-----------------------------------------------------------------------------

# Prompt for the verifier agent to collect key economic facts
VERIFIER_PROMPT = """Verify and provide the following specific economic facts about India's current situation:

1. Current GNI per capita (nominal): What is India's most recent GNI per capita in nominal terms (in USD)?

2. Current GNI per capita (PPP): What is India's most recent GNI per capita in PPP terms (in USD)?

3. Projected nominal growth rate: What is India's projected annual GNI per capita (nominal) growth rate over the next decade according to IMF/World Bank forecasts?

4. Projected PPP growth rate: What is India's projected annual GNI per capita (PPP) growth rate over the next decade according to IMF/World Bank forecasts?

5. Demographic transition year: In what year is India's working-age population share projected to peak and begin declining?

Format your response as follows (with precise numbers):
Current GNI per capita (nominal): $X,XXX
Current GNI per capita (PPP): $X,XXX
Projected nominal growth rate: X.X%
Projected PPP growth rate: X.X% 
Demographic transition year: 20XX

Do not include any additional commentary, analysis, or explanation. Provide only the facts in exactly the format specified."""

# Main question about India's economic future
# This template will be formatted with verified facts from the verifier agent
MAIN_QUESTION_TEMPLATE = """
As of {current_date}, can India become rich before it grows old?

Start your response with a single word — either 'Yes' or 'No'. Follow this immediately with a concise 1-2 sentence explanation of the core reason for this assessment.

Economic context:
- India's current GNI per capita (nominal): ${current_gni_nominal}
- India's current GNI per capita (PPP): ${current_gni_ppp}
- Rich country threshold: ${threshold_nominal} nominal per capita / ${threshold_ppp} PPP per capita
- Years until demographic peak: {years_to_peak} years (peak in {peak_year})
- Required annual growth rate (nominal): {required_rate_nominal:.1f}%
- Required annual growth rate (PPP): {required_rate_ppp:.1f}%
- Projected actual growth rate (nominal): {projected_rate_nominal:.1f}%
- Projected actual growth rate (PPP): {projected_rate_ppp:.1f}%

Focus on:
- Concrete shortfalls or advantages in current economic indicators
- Specific structural constraints or enablers
- Demographic timeline implications
- Quantified policy impacts

Conclude with direct data-backed assessment supported by specific statistics.
Use precise numbers and percentages throughout your analysis.
Include citations to relevant sources for all major claims.
"""

# Question about key influencers
PEOPLE_QUESTION = """Identify the top 5 individuals most responsible for shaping whether India becomes rich before it grows old.

Base your selection on this impact framework:
- Scope of authority (30%): Position power and decision-making control
- Proximity to binding constraints (25%): Influence over key economic bottlenecks 
- Scale of impact (20%): How many people/sectors their decisions affect
- Implementation power (15%): Ability to execute, not just decide
- Recent track record (10%): Demonstrated results in the past 2-3 years

Selection rules:
- ONLY include individuals CURRENTLY holding positions of power (as of 2025)
- EXCLUDE all former officials, even if they previously held influential positions
- For example, Shaktikanta Das is no longer the RBI Governor. As of March 2025, Sanjay Malhotra is the current governor. So, only include him if you consider the RBI Governor a key individual.
- VERIFY each individual's current role and position using the most recent sources
- For each person, confirm their exact current title before including them
- Double-check official government websites to ensure information is up-to-date
- If unsure about whether someone currently holds a position, DO NOT include them
- Cross-verify each role with at least two independent sources
- Prefer sources updated within the last month for the most recent data

Output format for each person:

<strong>[Full Name] ([Current Role, with title verified as of 2025])</strong>
[1–2 sentences with specific economic data explaining their impact on India's economic trajectory. Ensure explanations are crisp but include all relevant data.]

For each person's assessment, base justification on:
- Specific economic indicators showing impact of their policies
- Quantified results of their initiatives
- Direct evidence connecting their actions to growth metrics
"""

# Question about what's going wrong (for "No" answer)
WHATS_WRONG_QUESTION = """List 5 key statistical gaps preventing India from becoming rich before it grows old.

For each factor:
- Present as a direct data point showing the shortfall (e.g., "Manufacturing contributes only X% to GDP versus Y% needed")
- Include specific percentages, rates, or monetary figures
- Compare current metrics to required thresholds
- Cite the economic source of each data point

Focus on quantifiable gaps in:
- Growth rate differentials
- Job creation metrics
- Productivity measurements
- Infrastructure development percentages
- Human capital indicators

Present as concise, data-focused bullet points with no introductory text.
Each point should directly highlight a specific statistical shortfall with precise numbers.
"""

# Question about what's going right (for "Yes" answer)
WHATS_RIGHT_QUESTION = """List 5 key economic metrics indicating India will become rich before it grows old.

For each factor:
- Present a specific economic indicator with precise figures
- Include exact percentages, growth rates, or monetary values
- Compare to historical baselines or international benchmarks
- Cite the economic source of the data point

Focus on quantifiable strengths in:
- Growth rate metrics
- Productivity indicators
- Demographic dividend utilization data
- Infrastructure development percentages
- Human capital measurements

Present as concise, data-focused bullet points with no introductory text.
Each point should directly present a specific statistical strength with precise numbers.
"""

# Prompt for the review agent to evaluate generated content
REVIEW_PROMPT_TEMPLATE = """
Review the following economic analysis on whether India can become rich before it grows old:

===== CONTENT TO REVIEW =====
{content}
===== END CONTENT =====

Perform a structured evaluation of this content on the following dimensions:

1. Factual Consistency:
   - Are all numerical claims consistent with each other?
   - Are the economic growth projections logically sound?
   - Is the conclusion supported by the preceding analysis?

2. Evidential Support:
   - Is each major claim adequately supported by specific data or examples?
   - Are there any assertions that need stronger evidential backing?

3. Logical Flow:
   - Does the analysis follow a clear logical progression?
   - Are there any gaps in reasoning or unexplained leaps?

4. Precision:
   - Are economic terms used correctly and precisely?
   - Are there vague claims that could be made more specific?

5. Clarity:
   - Would a non-economist understand the key points?
   - Are there any passages that could be clearer or more concise?

For each dimension, provide:
- A score from 1-5 (where 5 is excellent)
- Specific examples of strengths and weaknesses
- Actionable suggestions for improvement

Format your response as a structured evaluation with separate sections for each dimension, followed by an overall assessment.""" 