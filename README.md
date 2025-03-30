# CanIndiaBecomeRichBeforeItGrowsOld.com

**Can India become rich before it grows old?** This repository powers a GitHub Pages website that attempts to answer this question with daily updates. The site uses the [Perplexity AI](https://www.perplexity.ai) API to fetch analysis and updates the content either daily or when economic facts need verification.

## How It Works

This project uses a streamlined approach with data verification to generate high-quality content:

- **Economic Facts Verification:** Runs every 30+ days to check and update economic facts
- **Content Generation:** Runs daily to generate the main content based on verified facts
- **Optional Review:** Quality checker that can be enabled with `--review` flag

The system operates with these key features:

- **Minimized API Calls:** Verification only runs every 30+ days to reduce API costs
- **Data Validation:** Sanity checks are performed to ensure quality
- **Fallback Mechanisms:** Default values are used if data is missing or invalid
- **Simplified Structure:** All prompts and constants in a single file for easy management

## Repository Layout

```
project/
├── prompt.py            # All prompts and constants in a single file
├── utils/               # Utility functions and data models
├── template.html        # HTML template for the page
├── update_answer.py     # Main script for generating content
├── impact-framework.html # Methodology explanation page
└── .github/workflows/update.yml # GitHub Actions workflow
```

## Key Components

- **prompt.py:** Contains all prompts, constants, and default values in one place
- **update_answer.py:** Main script that orchestrates the content generation and updates the page
- **template.html:** HTML template with placeholders for dynamic content
- **.github/workflows/update.yml:** Workflow configuration for GitHub Actions with environment variables

## Setup & Configuration

To set up this project on your own repository or to contribute, follow these steps:

1. **Perplexity API Key:** Obtain an API key from Perplexity AI and add it as a secret in your GitHub repository settings. Name the secret `PERPLEXITY_API_KEY`.
2. **GitHub Pages:** Enable GitHub Pages on the repository. If using a custom domain, ensure you add a DNS CNAME record.
3. **Environment Variables:** The workflow uses the `LAST_VERIFIER_RUN` environment variable to track when the last verification was run. You can adjust this in `.github/workflows/update.yml`.
4. **Review Option:** Set `RUN_REVIEW: "true"` in the workflow file to enable the review functionality.

## Local Development

To run and test the project locally:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install requests python-dotenv markdown2
   ```
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and add your Perplexity API key
5. Run the update script:
   ```bash
   python update_answer.py
   ```
   
   To run with the review option enabled:
   ```bash
   python update_answer.py --review
   ```

The `.env` file is ignored by git (see `.gitignore`), so your API key will remain secure.

## Contributing

Contributions are welcome! If you have ideas to improve the analysis, add more sections, or refine the prompts, feel free to open an issue or pull request. When contributing, please keep the code simple and well-documented for ease of maintenance.

---

*A data-verified analysis of India's economic and demographic future, updated daily.*
