from openai import OpenAI

client = OpenAI()

def generate_summary(chunks):
    full_text = " ".join(chunks)[:12000]
    prompt = f"""
    Summarize the following research paper in 4 sections:
    - Abstract
    - Methods
    - Results
    - Insights
    Paper Text:
    {full_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    summary_text = response.choices[0].message.content
    return {
        "abstract": summary_text.split("Methods:")[0].strip(),
        "methods": "Extracted methods section (demo)",
        "results": "Extracted results section (demo)",
        "insights": "Extracted insights (demo)"
    }
