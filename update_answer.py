#!/usr/bin/env python3
"""
Simplified data-verification pipeline for the
"Can India Become Rich Before It Grows Old?" project.

This script orchestrates:
1. Economic fact verification (runs every 30+ days)
2. Content generation (runs daily)
3. Optional content review (runs on demand)

Usage:
    python update_answer.py [--review]
    
    --review: Optional flag to run content review
"""

import os
import sys
import re
import argparse
import logging
import requests
import markdown2
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("update.log", "a")
    ]
)

# Import all prompts and constants
from prompt import (
    # Constants
    HIGH_INCOME_THRESHOLD_NOMINAL, 
    HIGH_INCOME_THRESHOLD_PPP,
    DEMOGRAPHIC_PEAK_YEAR,
    VERIFIER_INTERVAL_DAYS,
    API_URL,
    API_MODEL,
    API_TEMPERATURE,
    API_TOP_P,
    PRIMARY_SOURCES,
    DEFAULT_FACTS,
    YEARS_TO_DEMOGRAPHIC_PEAK,
    
    # System prompts
    VERIFIER_SYSTEM_PROMPT,
    DATA_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    
    # User prompts
    VERIFIER_PROMPT,
    MAIN_QUESTION_TEMPLATE,
    PEOPLE_QUESTION,
    WHATS_WRONG_QUESTION,
    WHATS_RIGHT_QUESTION,
    REVIEW_PROMPT_TEMPLATE
)

# Define data model class
class PageContent:
    """
    Data model for the content on the page.
    """
    def __init__(
        self,
        answer: str,
        explanation: str,
        people: List[str],
        factors: List[str],
        sources: List[Tuple[str, str]],
        last_updated: str,
        section_heading: str,
        repo_url: str = ""
    ):
        self.answer = answer
        self.explanation = explanation
        self.people = people
        self.factors = factors
        self.sources = sources
        self.last_updated = last_updated
        self.section_heading = section_heading
        self.repo_url = repo_url

class Feedback:
    """
    Data model for content review feedback.
    """
    def __init__(
        self,
        overall_score: float,
        factual_score: float,
        evidence_score: float,
        logic_score: float,
        precision_score: float,
        clarity_score: float,
        overall_feedback: str
    ):
        self.overall_score = overall_score
        self.factual_score = factual_score
        self.evidence_score = evidence_score
        self.logic_score = logic_score
        self.precision_score = precision_score
        self.clarity_score = clarity_score
        self.overall_feedback = overall_feedback

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the 'Can India Become Rich Before It Grows Old?' page."
    )
    parser.add_argument("--review", action="store_true", help="Run content review")
    return parser.parse_args()

# API Utility Functions

def ask_perplexity(
    prompt: str, 
    api_key: str, 
    system_prompt: str, 
    temperature: Optional[float] = None
) -> Tuple[str, List[str]]:
    """
    Ask Perplexity API a question and return the response and citations.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    # Append primary sources to the prompt
    full_prompt = prompt + "\n\nPrimary Sources:\n" + "\n".join(PRIMARY_SOURCES)
    
    # Use provided temperature or fall back to constant
    actual_temperature = temperature if temperature is not None else API_TEMPERATURE
    
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": actual_temperature,
        "top_p": API_TOP_P,
        "return_images": False,
        "return_related_questions": False,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract the main response and citations
        answer = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        
        return answer, citations
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Perplexity API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        raise

def should_run_verifier() -> bool:
    """
    Determine if the verifier should run based on the last run date.
    """
    last_run_str = os.environ.get("LAST_VERIFIER_RUN", "")
    
    # If the environment variable isn't set or is empty, run the verifier
    if not last_run_str:
        logging.info("No LAST_VERIFIER_RUN found, running verifier")
        return True
    
    try:
        # Parse the date from the environment variable (expecting ISO format)
        last_run_date = datetime.fromisoformat(last_run_str)
        today = datetime.utcnow()
        days_since_last_run = (today - last_run_date).days
        
        # Run if it's been more than VERIFIER_INTERVAL_DAYS since the last run
        if days_since_last_run >= VERIFIER_INTERVAL_DAYS:
            logging.info(f"Last verifier run was {days_since_last_run} days ago, running verifier")
            return True
        else:
            logging.info(f"Last verifier run was only {days_since_last_run} days ago, skipping verifier")
            return False
    except ValueError:
        # If there's any issue parsing the date, run the verifier to be safe
        logging.warning(f"Failed to parse LAST_VERIFIER_RUN value: {last_run_str}, running verifier")
        return True

def parse_verified_facts(response: str) -> Dict[str, Any]:
    """
    Parse the AI response to extract structured economic data.
    """
    facts = {}
    
    # Log the raw API response for debugging
    logging.debug(f"Raw API response: {response}")
    
    # Extract India's current GNI per capita (nominal)
    gni_nominal_match = re.search(r'Current GNI per capita \(nominal\):?\s*\$?([0-9,]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
    if gni_nominal_match:
        gni_value = gni_nominal_match.group(1).replace(',', '')
        facts["current_gni_per_capita_nominal"] = float(gni_value)
    else:
        logging.warning("Current GNI per capita (nominal) not found in response, will use default")
    
    # Extract India's current GNI per capita (PPP)
    gni_ppp_match = re.search(r'Current GNI per capita \(PPP\):?\s*\$?([0-9,]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
    if gni_ppp_match:
        gni_value = gni_ppp_match.group(1).replace(',', '')
        facts["current_gni_per_capita_ppp"] = float(gni_value)
    else:
        logging.warning("Current GNI per capita (PPP) not found in response, will use default")
    
    # Extract projected nominal growth rate
    nominal_growth_match = re.search(r'Projected nominal growth rate:?\s*([0-9]+(?:\.[0-9]+)?)%?', response, re.IGNORECASE)
    if nominal_growth_match:
        facts["projected_growth_rate_nominal"] = float(nominal_growth_match.group(1))
    else:
        logging.warning("Projected nominal growth rate not found in response, will use default")

    # Extract projected PPP growth rate
    ppp_growth_match = re.search(r'Projected PPP growth rate:?\s*([0-9]+(?:\.[0-9]+)?)%?', response, re.IGNORECASE)
    if ppp_growth_match:
        facts["projected_growth_rate_ppp"] = float(ppp_growth_match.group(1))
    else:
        logging.warning("Projected PPP growth rate not found in response, will use default")
    
    # Extract demographic timeline
    demographic_match = re.search(r'Demographic transition year:?\s*(\d{4})', response, re.IGNORECASE)
    if demographic_match:
        facts["demographic_peak_year"] = int(demographic_match.group(1))
    else:
        logging.warning("Demographic peak year not found in response, will use default")
    
    # Calculate years to demographic peak
    current_year = datetime.utcnow().year
    demographic_year = facts.get("demographic_peak_year", DEMOGRAPHIC_PEAK_YEAR)
    facts["years_to_demographic_peak"] = demographic_year - current_year
    
    # Calculate required growth rates using the formula
    def calculate_growth_rate(current_value, target_value, years):
        """Calculate compound annual growth rate required to reach target from current value in given years"""
        return ((target_value / current_value) ** (1/years) - 1) * 100
    
    # Calculate required nominal growth rate
    if "current_gni_per_capita_nominal" in facts:
        facts["required_growth_rate_nominal"] = calculate_growth_rate(
            facts["current_gni_per_capita_nominal"], 
            HIGH_INCOME_THRESHOLD_NOMINAL, 
            facts["years_to_demographic_peak"]
        )
    
    # Calculate required PPP growth rate
    if "current_gni_per_capita_ppp" in facts:
        facts["required_growth_rate_ppp"] = calculate_growth_rate(
            facts["current_gni_per_capita_ppp"], 
            HIGH_INCOME_THRESHOLD_PPP, 
            facts["years_to_demographic_peak"]
        )
    
    # Add the high-income thresholds
    facts["high_income_threshold_nominal"] = HIGH_INCOME_THRESHOLD_NOMINAL
    facts["high_income_threshold_ppp"] = HIGH_INCOME_THRESHOLD_PPP
    
    # Log the found facts for debugging purposes
    logging.info(f"Parsed facts from API response: {facts}")
    
    return facts

def get_latest_verified_facts(api_key: str) -> Dict[str, Any]:
    """
    Get the most up-to-date economic facts, either by verification or using defaults.
    """
    # Start with default values
    merged_facts = DEFAULT_FACTS.copy()
    
    if should_run_verifier():
        logging.info("Starting economic facts verification")
        try:
            # Get the verified facts
            response, _ = ask_perplexity(VERIFIER_PROMPT, api_key, VERIFIER_SYSTEM_PROMPT, temperature=0.1)
            
            # Parse the response to extract facts
            verified_facts = parse_verified_facts(response)
            
            # Update last run timestamp in environment for future reference
            current_time = datetime.utcnow().isoformat()
            os.environ["LAST_VERIFIER_RUN"] = current_time
            
            # Write the date to the file (for GitHub Actions to track)
            try:
                with open("last_verification.txt", "w") as f:
                    f.write(current_time)
                logging.info(f"Updated last verification date to {current_time}")
            except Exception as e:
                logging.warning(f"Could not write to last_verification.txt: {e}")
            
            # Check for required fields and log if any are missing
            required_fields = [
                "current_gni_per_capita_nominal", "current_gni_per_capita_ppp",
                "projected_growth_rate_nominal", "projected_growth_rate_ppp",
                "demographic_peak_year"
            ]
            missing_fields = [field for field in required_fields if field not in verified_facts]
            
            if missing_fields:
                logging.warning(f"Verification succeeded but missing required fields: {missing_fields}")
                logging.warning(f"Using default values for missing fields")
            
            # Update merged_facts with any values that were successfully verified
            if verified_facts:
                merged_facts.update(verified_facts)
            
            logging.info(f"Economic facts verification completed successfully")
        except Exception as e:
            logging.error(f"Error verifying economic facts: {e}")
            # Continue with default values
        
        logging.info(f"Final facts (with defaults for missing fields): {merged_facts}")
    else:
        logging.info(f"Using default facts: {DEFAULT_FACTS}")
    
    return merged_facts

# Helper functions for content generation

def parse_list_items(text: str) -> List[str]:
    """
    Parse text into list items, cleaning bullet points and applying Markdown.
    """
    # Split into paragraphs or numbered points
    # Look for paragraphs separated by blank lines, or numbered points
    items = []
    
    # First try to identify numbered points in the text
    numbered_points = re.findall(r'(?:\d+\.|\*|\-)\s*(.*?)(?=\n\n|\n(?:\d+\.|\*|\-)|\Z)', text, re.DOTALL)
    
    # If we found clear numbered points, process them
    if numbered_points and len(numbered_points) >= 3:
        for point in numbered_points:
            # Clean the point
            cleaned = point.strip()
            if not cleaned:
                continue
                
            # Remove any remaining numbering or bullets
            cleaned = re.sub(r'^(?:\d+\.|\*|\-)\s*', '', cleaned)
            
            # Remove any "Factor:" prefix or leading colons
            cleaned = re.sub(r'^(?:Factor:)?\s*', '', cleaned)
            cleaned = re.sub(r'^:\s*', '', cleaned)
            
            # Convert to HTML
            html = markdown2.markdown(cleaned).strip()
            # Remove wrapping <p> tags if present
            html = re.sub(r'^<p>(.*)</p>$', r'\1', html)
            
            items.append(html)
    else:
        # Fall back to paragraph-based parsing
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Skip the introductory text if present
            if "Here is my ranked list" in para or "Here are" in para:
                continue
                
            # Clean up the paragraph
            lines = [line.strip() for line in para.splitlines() if line.strip()]
            if not lines:
                continue
                
            # Process each line to remove lead-ins like bullets, numbers, and colons
            cleaned_lines = []
            for line in lines:
                # Remove leading bullet characters, numbers, and extra whitespace
                cleaned = re.sub(r'^[-•*\d\.]+\s*', '', line).strip()
                
                # More aggressive pattern to catch any variations of leading colons
                # This will remove colons at the beginning of the line or after a word/phrase followed by a colon
                cleaned = re.sub(r'^([^:]*:)?\s*', '', cleaned).strip()
                
                if cleaned:
                    cleaned_lines.append(cleaned)
                    
            if cleaned_lines:
                # Join the lines and convert to HTML
                html = markdown2.markdown('\n'.join(cleaned_lines)).strip()
                # Remove wrapping <p> tags if present
                html = re.sub(r'^<p>(.*)</p>$', r'\1', html)
                
                # Final check to remove any remaining colons at the beginning
                html = re.sub(r'^:\s*', '', html).strip()
                
                # Check if this is a long paragraph with multiple sentences
                # If so, try to split it into separate factors
                if len(html) > 200 and html.count('.') > 2:
                    # Split by sentences, being careful with abbreviations and numbers
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', html)
                    for sentence in sentences:
                        if sentence.strip():
                            items.append(sentence.strip())
                else:
                    items.append(html)
    
    return items

def parse_people_items(text: str) -> List[str]:
    """
    Parse people list items, ensuring proper formatting with name/title in bold.
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    items = []
    
    for para in paragraphs:
        # Skip any introductory text
        if "Based on the impact framework" in para or "Here is my ranked list" in para or "Here are" in para or "Following individuals" in para or "Key figures" in para:
            continue
            
        # Remove any OL tags or numbering that might be in the text
        para = re.sub(r'<ol[^>]*>|</ol>|<li>|</li>|\d+\.\s*', '', para).strip()
        
        # Look for name and title in strong tags
        name_match = re.search(r'<strong>(.*?)</strong>', para)
        
        if name_match:
            # Extract the name/title part that should be in bold
            name_part = name_match.group(1)
            
            # Get the explanation by removing the name part
            explanation = para.replace(name_match.group(0), '').strip()
            
            # Clean up the explanation (remove extra whitespace, etc.)
            explanation = re.sub(r'\s+', ' ', explanation).strip()
            
            # Format in the desired order: name in bold, then explanation
            formatted_item = f'<strong>{name_part}</strong> {explanation}'
            items.append(formatted_item)
        else:
            # If parsing fails, just clean up and use the original text
            html = markdown2.markdown(para.strip()).strip()
            # Remove wrapping <p> tags if present
            html = re.sub(r'^<p>(.*)</p>$', r'\1', html)
            
            # Skip any items that are likely to be introductory text
            if not re.search(r'(framework|identified|following|key figures)', html, re.IGNORECASE):
                items.append(html)
    
    return items

def parse_citations(citations: List[str]) -> List[Tuple[str, str]]:
    """
    Parse citation strings into (url, label) tuples.
    """
    parsed = []
    for cit in citations:
        m = re.match(r'【\d+†(.*?)†(.*?)】', cit)
        if m:
            url = m.group(1)
            site_label = m.group(2) or m.group(1)
            parsed.append((url, site_label))
    return parsed

def remove_citations(text: str) -> str:
    """
    Remove all [] citations from the text.
    """
    return re.sub(r'\[.*?\]', '', text)

def validate_yes_no_response(response: str) -> str:
    """
    Validate a Yes/No response and return a standardized value.
    """
    if response.lower() in ["yes", "no"]:
        return response.capitalize()
    return "Unknown"

def validate_people_list(people: List[str]) -> List[str]:
    """
    Validate the list of people, ensuring it retains the original ranking order
    based on the impact framework.
    """
    # Filter out empty items
    people = [p for p in people if p.strip()]
    
    # If we have less than 3 items, log a warning
    if len(people) < 3:
        logging.warning(f"Only {len(people)} people found, which is below the 3 minimum expected")
    
    # Return maximum 5 items, maintaining the original order
    # (ordering is assumed to be done by the model based on the impact framework)
    return people[:5]

def validate_actions_list(actions: List[str]) -> List[str]:
    """
    Validate the list of actions, ensuring it has 5 items if possible.
    """
    # Filter out empty items
    actions = [a for a in actions if a.strip()]
    
    # If we have less than 3 items, log a warning
    if len(actions) < 3:
        logging.warning(f"Only {len(actions)} actions found, which is below the 3 minimum expected")
    
    # Return maximum 5 items
    return actions[:5]

def generate_content(api_key: str, verified_facts: Dict[str, Any]) -> PageContent:
    """
    Generate the main content based on verified facts.
    """
    logging.info("Starting content generation")
    
    # Log the incoming verified facts
    logging.info(f"Using verified facts: {verified_facts}")

    # Format the main question with verified facts
    formatted_main_question = MAIN_QUESTION_TEMPLATE.format(
        current_date=datetime.utcnow().strftime("%B %d, %Y"),
        current_gni_nominal=f"{verified_facts['current_gni_per_capita_nominal']:,.1f}",
        current_gni_ppp=f"{verified_facts['current_gni_per_capita_ppp']:,.1f}",
        threshold_nominal=f"{verified_facts['high_income_threshold_nominal']:,.0f}",
        threshold_ppp=f"{verified_facts['high_income_threshold_ppp']:,.0f}",
        peak_year=verified_facts['demographic_peak_year'],
        years_to_peak=verified_facts['years_to_demographic_peak'],
        required_rate_nominal=verified_facts['required_growth_rate_nominal'],
        required_rate_ppp=verified_facts['required_growth_rate_ppp'],
        projected_rate_nominal=verified_facts['projected_growth_rate_nominal'],
        projected_rate_ppp=verified_facts['projected_growth_rate_ppp']
    )
    
    # Get the main answer to determine if India can become rich before growing old
    answer_text, answer_citations = ask_perplexity(
        formatted_main_question, 
        api_key,
        DATA_SYSTEM_PROMPT
    )
    
    # Parse answer into headline (Yes/No) and explanation
    match = re.match(r'^(Yes|No)\b[\.:,\-]?\s*(.*)', answer_text.strip(), flags=re.IGNORECASE)
    if match:
        answer = match.group(1).capitalize()
        explanation = markdown2.markdown(remove_citations(match.group(2).strip()))
    else:
        # If we can't parse the answer format, default to Unknown
        answer = "Unknown"
        explanation = markdown2.markdown(remove_citations(answer_text.strip()))
    
    # Validate the answer
    answer = validate_yes_no_response(answer)
    
    # Get list of key people
    people_text, _ = ask_perplexity(PEOPLE_QUESTION, api_key, DATA_SYSTEM_PROMPT)
    people = parse_people_items(remove_citations(people_text))
    people = validate_people_list(people)
    
    # Based on the Yes/No answer, ask about what's going right or wrong
    if answer.lower() == "yes":
        factors_text, _ = ask_perplexity(WHATS_RIGHT_QUESTION, api_key, DATA_SYSTEM_PROMPT)
        section_heading = "What Is Going Right"
    else:
        factors_text, _ = ask_perplexity(WHATS_WRONG_QUESTION, api_key, DATA_SYSTEM_PROMPT)
        section_heading = "What Is Going Wrong"
    
    factors = parse_list_items(remove_citations(factors_text))
    factors = validate_actions_list(factors)
    
    # Parse citations from the main answer (which contained the factual claims)
    sources = parse_citations(answer_citations)
    
    # Get repository URL if in GitHub Actions
    repo_url = os.environ.get("GITHUB_REPOSITORY", "")
    if repo_url:
        repo_url = f"https://github.com/{repo_url}"
    
    # Create and return the page content object
    content = PageContent(
        answer=answer,
        explanation=explanation,
        people=people,
        factors=factors,
        sources=sources,
        last_updated=datetime.utcnow().strftime("%B %d, %Y"),
        section_heading=section_heading,
        repo_url=repo_url
    )
    
    logging.info(f"Content generation completed successfully with answer: {answer}")
    return content

def review_content(api_key: str, content: PageContent) -> Optional[Feedback]:
    """
    Review the generated content for quality.
    """
    logging.info("Starting content review")
    
    # Format the content as a string for review - avoiding f-string issues with backslashes
    factors_list = "\n".join(["- " + item for item in content.factors])
    people_list = "\n".join(["- " + item for item in content.people])
    
    content_str = f"""
Answer: {content.answer}

Explanation:
{content.explanation}

{content.section_heading}:
{factors_list}

Key Individuals Responsible:
{people_list}
"""
    
    # Format the review prompt
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(content=content_str)
    
    try:
        # Get the review
        review_text, _ = ask_perplexity(review_prompt, api_key, REVIEW_SYSTEM_PROMPT)
        
        # Parse the scores (simple regex to find scores 1-5 after each dimension's heading)
        scores = {}
        dimensions = ["Factual Consistency", "Evidential Support", "Logical Flow", "Precision", "Clarity"]
        
        for dim in dimensions:
            match = re.search(rf"{dim}[^\d]*(\d+(?:\.\d+)?)/5", review_text, re.IGNORECASE)
            scores[dim] = float(match.group(1)) if match else 3.0  # Default to 3 if not found
            
        # Parse the overall assessment
        overall_match = re.search(r"Overall[^:]*:.*?(\d+(?:\.\d+)?)/5", review_text, re.IGNORECASE | re.DOTALL)
        overall_score = float(overall_match.group(1)) if overall_match else sum(scores.values()) / len(scores)
        
        # Extract overall feedback
        feedback_match = re.search(r"Overall[^:]*:(.*?)(?=\n\n|$)", review_text, re.IGNORECASE | re.DOTALL)
        overall_feedback = feedback_match.group(1).strip() if feedback_match else "No overall feedback provided."
        
        # Create and return the feedback object
        feedback = Feedback(
            overall_score=overall_score,
            factual_score=scores.get("Factual Consistency", 3.0),
            evidence_score=scores.get("Evidential Support", 3.0),
            logic_score=scores.get("Logical Flow", 3.0),
            precision_score=scores.get("Precision", 3.0),
            clarity_score=scores.get("Clarity", 3.0),
            overall_feedback=overall_feedback
        )
        
        logging.info(f"Content review completed with overall score: {feedback.overall_score:.1f}/5.0")
        return feedback
    except Exception as e:
        logging.error(f"Error reviewing content: {e}")
        return None

def generate_html(content: PageContent) -> str:
    """
    Generate the final HTML content from structured data.
    """
    logging.info("Generating HTML content")
    
    # Read template
    try:
        with open("template.html", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        logging.error("template.html not found, using minimal fallback template")
        template = get_fallback_template()
    
    # Determine the answer suffix
    answer_suffix = " not" if content.answer.lower() == "no" else ""
    
    # Generate HTML for lists - ensuring people are in a clean bulleted list
    people_html = ""
    for i, item in enumerate(content.people):
        # Format each person as a clean list item
        people_html += f"<li>{item}</li>\n"
    
    # Clean any residual leading colons in factors and generate HTML
    cleaned_factors = []
    for factor in content.factors:
        # More thorough cleanup to ensure no leading colons or "Factor:" patterns
        # This will catch patterns like "Factor:", ":", or any word followed by a colon
        factor = re.sub(r'^([^:]*:)?\s*', '', factor).strip()
        # Double-check to catch any remaining isolated colons
        factor = re.sub(r'^:\s*', '', factor).strip()
        cleaned_factors.append(factor)
    
    factors_html = "".join(f"<li>{factor}</li>\n" for factor in cleaned_factors)
    
    # Ensure sources are properly formatted
    sources_html = "".join(
        f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a></li>\n'
        for url, label in content.sources
    )
    
    # If no sources were found, use PRIMARY_SOURCES as default
    if not content.sources:
        default_sources_html = []
        for source in PRIMARY_SOURCES:
            parts = source.split(": ")
            if len(parts) == 2:
                name, url = parts
                default_sources_html.append(f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{name}</a></li>')
            else:
                default_sources_html.append(f'<li>{source}</li>')
        
        sources_html = "\n".join(default_sources_html)
    
    # Use default repo URL if none is provided
    repo_url = content.repo_url or "https://github.com/ajayrmk/CanIndiaBecomeRichBeforeItGrowsOld"
    
    # Format the project info
    project_info = f"""Updated daily using the Perplexity API — last updated on {content.last_updated}.<br>
If you find this project useful, you can <a href="https://buymeacoffee.com/ajayram" target="_blank">support it here</a> to help offset API usage costs.<br><br>
View the <a href="{repo_url}" target="_blank">source code on GitHub</a>."""
    
    # Replace placeholders in template
    html = template
    html = html.replace("{{answer}}", content.answer)
    html = html.replace("{{answer_suffix}}", answer_suffix)
    html = html.replace("{{date}}", content.last_updated)
    html = html.replace("{{why}}", content.explanation)
    html = html.replace("{{section_heading}}", content.section_heading)
    html = html.replace("{{factors}}", factors_html)
    html = html.replace("{{people}}", people_html)
    html = html.replace("{{sources}}", sources_html)
    html = html.replace("{{project_info}}", project_info)
    
    logging.info("HTML generation complete")
    return html

def get_fallback_template() -> str:
    """
    Provide a minimal fallback template in case the main template is missing.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Can India Become Rich Before It Grows Old?</title>
  <meta property="og:title" content="Can India Become Rich Before It Grows Old?" />
  <meta property="og:description" content="{{answer}}, it will{{answer_suffix}} as of {{date}}." />
  <meta property="og:image" content="favicon.png" />
  <meta property="og:url" content="https://canindiabecomerichbeforeitgrowsold.com/" />
  <meta name="twitter:card" content="summary_large_image" />
  <style>
    body { font-family: Georgia, serif; margin: 0; padding: 0; line-height: 1.6; text-align: left; background-color: #f9f8f6; }
    main { max-width: 750px; margin: auto; padding: 20px; }
    header { margin-bottom: 2em; text-align: left; }
    .question { font-style: italic; color: #666; margin-bottom: 0.5em; font-size: 1.1em; }
    h1 { font-size: 3.5em; margin: 0.5em 0; font-weight: bold; }
    .subtitle { font-size: 1.2em; color: #666; margin-bottom: 1.5em; }
    h2 { font-size: 1.5em; margin-top: 2em; margin-bottom: 0.5em; border-left: 4px solid #ccc; padding-left: 0.5em; }
    ul, ol { margin: 0; padding-left: 1.5em; }
    li { margin-bottom: 1em; }
    li strong { display: inline; margin-right: 0.3em; }
    footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #ccc; font-size: 0.9em; color: #555; }
    a { color: #0366d6; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="question">Can India become rich before it grows old?</div>
      <h1 id="answer">{{answer}}</h1>
      <div class="subtitle">(as of {{date}})</div>
    </header>
    <div class="explanation">{{why}}</div>

    <!-- Factors section -->
    <h2>{{section_heading}}</h2>
    <ol>
      {{factors}}
    </ol>

    <!-- Key people section -->
    <h2>Key Individuals Responsible</h2>
    <ol>
      {{people}}
    </ol>

    <!-- Sources section -->
    <h2>Sources</h2>
    <ol>
      {{sources}}
    </ol>

    <!-- Project info section -->
    <footer>
      <p>{{project_info}}</p>
      <p><a href="impact-framework.html">See full methodology for ranking key individuals</a></p>
    </footer>
  </main>
</body>
</html>"""

def write_html_to_file(html_content: str, output_file: str = "index.html") -> bool:
    """
    Write the generated HTML to the output file.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if content has changed
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing_content = f.read()
            
            if existing_content.strip() == html_content.strip():
                logging.info(f"No changes in content. Skipping update to {output_file}.")
                return True
        
        # Write new content
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logging.info(f"{output_file} updated with new content.")
        return True
        
    except Exception as e:
        logging.error(f"Error writing HTML to {output_file}: {e}")
        return False

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Get the API key
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logging.error("PERPLEXITY_API_KEY environment variable is not set.")
        sys.exit(1)
    
    logging.info("Starting pipeline")
    
    try:
        # Step 1: Get verified facts
        verified_facts = get_latest_verified_facts(api_key)
        
        # Step 2: Generate content using verified facts
        content = generate_content(api_key, verified_facts)
        
        # Step 3: Optionally review content
        if args.review:
            feedback = review_content(api_key, content)
            if feedback and feedback.overall_score < 3.5:
                logging.warning(
                    f"Content quality below threshold: {feedback.overall_score:.1f}/5.0\n"
                    f"Overall feedback: {feedback.overall_feedback}"
                )
        
        # Step 4: Generate HTML
        html_content = generate_html(content)
        
        # Step 5: Write HTML to file
        success = write_html_to_file(html_content)
        if success:
            logging.info("Pipeline completed successfully")
        else:
            logging.error("Failed to write HTML to file")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 