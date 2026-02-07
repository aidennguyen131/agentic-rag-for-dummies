import sys
import os
import base64
from pathlib import Path

# Add project directory to path
# Add project directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(base_dir, "project")
sys.path.append(project_path)

from core.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import config

class CreativeAgent:
    def __init__(self):
        print("Initializing Creative Agent...")
        self.rag = RAGSystem()
        self.rag.initialize()
        
        # Initialize Vision LLM (OpenAI GPT-4o-mini is cost effective and good)
        self.vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500)
        self.creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path):
        print(f"Analyzing image: {image_path}...")
        base64_image = self.encode_image(image_path)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this image for a T-shirt design. Describe the Subject, Art Style, Colors, Mood, and Key Visual Elements. Be concise but descriptive."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        response = self.vision_llm.invoke([message])
        return response.content

    def retrieve_ideas(self, description, user_instruction=None):
        print("Retrieving related concepts from RAG...")
        # Formulate a query for the RAG agent
        query = f"Based on this visual description: '{description}', suggest relevant funny T-shirt keywords, puns, or style concepts from the database."
        
        if user_instruction:
            query += f"\n\nUSER COMMAND: {user_instruction}\n(Prioritize keywords related to this command)"
            
        # Invoke the Agentic RAG graph
        # The graph state expects 'messages'
        inputs = {"messages": [HumanMessage(content=query)]}
        
        # We need to run the graph. 
        # The graph returns a dict with 'messages' and 'agent_answers'
        result = self.rag.agent_graph.invoke(inputs, config=self.rag.get_config())
        
        # Extract the final answer (usually the last AIMessage)
        final_message = result["messages"][-1]
        return final_message.content

    def mix_and_create(self, description, rag_ideas, user_instruction=None):
        print("Mixing ideas...")
        from pydantic import BaseModel, Field
        from typing import List

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Detailed prompt for AI image generator")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Business logic or why this design works")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

        prompt = f"""
        You are a Creative Director for a POD (Print on Demand) T-shirt business.
        
        Original Image Analysis:
        {description}
        
        Inspiration from our Database (Trending Keywords/Styles):
        {rag_ideas}
        
        """
        
        if user_instruction:
            prompt += f"""
            IMMEDIATE USER INSTRUCTION:
            "{user_instruction}"
            (You MUST strictly follow this instruction. If it asks to change style, change style. If it asks for specific subject, ignore RAG data if it conflicts.)
            """
            
        prompt += """
        Task:
        Create 3 unique, creative T-shirt design concepts that **ACTIVELY MASH UP** the original subject with the retrieved keywords/styles.
        
        CRITICAL INSTRUCTION:
        - DO NOT just describe the original image in more detail.
        - DO NOT just add the style as a filter.
        - **COMBINE** concepts to create something new. (e.g., If Image is "Cat" and Keyword is "Ramen", make "Cat eating Ramen" or "Ramen made of Yarn").
        - Use **Surprise** and **Humor**.
        - Aim for "Visual Puns" or "Ironic Juxtapositions".
        
        Make them funny, catchy, and market-ready.
        """
        
        structured_llm = self.creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.concepts

    def run(self, image_path, user_instruction=None):
        # 1. Vision (Cache this if optimizing, but cheap enough to re-run or just pass description if we structured it better)
        # For simplicity, we re-analyze or assumed the caller might optimize. 
        # Actually, let's just re-run vision for now to be stateless-ish, or better: 
        # Ideally we pass the description in if we have it, but image_path is the interface.
        description = self.analyze_image(image_path)
        print(f"\n[Vision Analysis]\n{description}\n")
        
        # 2. RAG
        rag_ideas = self.retrieve_ideas(description, user_instruction)
        print(f"\n[RAG Suggestions]\n{rag_ideas}\n")
        
        # 3. Creative Mix
        final_concepts = self.mix_and_create(description, rag_ideas, user_instruction)
        print(f"\n[Final Concepts]\n{final_concepts}\n")
        
        # Return dict for API
        return {
            "vision_analysis": description,
            "rag_suggestions": rag_ideas,
            "concepts": [c.dict() for c in final_concepts]
        }

    def remix_concept(self, original_concept: dict):
        print(f"Remixing concept: {original_concept.get('title')}...")
        from pydantic import BaseModel, Field
        from typing import List

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Detailed prompt for AI image generator")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Why this variation works")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

        prompt = f"""
        You are a Creative Director for a POD T-shirt business.
        We have a successful design concept, and we want 3 DISTINCT variations or "remixes" of it.
        
        Original Concept:
        Title: {original_concept.get('title')}
        Visual: {original_concept.get('visual_prompt')}
        Caption: {original_concept.get('caption')}
        
        Task:
        Create 3 BOLD variations using **SCAMPER** techniques (Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse).
        
        CRITICAL RULES:
        1. **NO MORE DETAIL**: Do not just add adjectives or make the description longer.
        2. **CHANGE THE ANGLE**:
           - **Variation 1 (The Mashup)**: Combine the subject with a completely different object or hobby (e.g., Coffee, Gym, Gaming, Space).
           - **Variation 2 (The Role Reverse)**: Put the subject in an unexpected human situation or ironic context.
           - **Variation 3 (The Style Twist)**: Keep the subject but radically change the art style (e.g., from Vintage to Kawaii, or Realistic to Minimalist line art).
        
        3. **NEW CAPTIONS**: You MUST change the caption to match the new twist.
        """
        
        structured_llm = self.creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return [c.dict() for c in response.concepts]

if __name__ == "__main__":
    # Test with a dummy image path or ask user
    # Ideally checking if an image exists or creating a dummy
    agent = CreativeAgent()
    
    # Check for test image
    test_image = "test_image.jpg"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if os.path.exists(test_image):
        agent.run(test_image)
    else:
        print(f"Please provide an image path. Usage: python creative_agent.py <path_to_image>")
