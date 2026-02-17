from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
    template ="""
Summarize the research paper named {paper_input} with explanation style:{style_input} and output length as: {length_input}
1.Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2.Analogies:
 - Use relatable analogies to simplify complex ideas.
If certain information is missing, respond with "Information insufficient" instead of guessing.
""",
input_variables=["paper_input", "style_input", "length_input"]   
)
template.save("prompt_template.json")