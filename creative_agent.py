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
        
        from pydantic import BaseModel, Field
        from typing import List

        class VisionAnalysis(BaseModel):
            subject: str = Field(description="The main character or subject")
            action: str = Field(description="What the subject is doing")
            context: str = Field(description="Environment, background, or setting")
            art_style: str = Field(description="The visual style (e.g. vintage, vector, photo-real)")
            colors: str = Field(description="Dominant colors and palette description")
            mood: str = Field(description="The emotional tone (e.g. funny, scary, serious)")
            key_elements: List[str] = Field(description="List of other important visual elements")

        base64_image = self.encode_image(image_path)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this image for a T-shirt design. Extract the following specific details:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        # Use structured output
        structured_llm = self.vision_llm.with_structured_output(VisionAnalysis)
        response = structured_llm.invoke([message])
        
        # Return the dict, or handle it in run()
        # For compatibility with existing string-based flow, we might need to adjust,
        # but let's return the dict object so we can pass it to frontend.
        # However, mix_and_create expects a description string.
        # We should return the dict, and let caller handle it.
        # Wait, if we change return type, we break mix_and_create which expects a string description?
        # Let's check mix_and_create usage. It uses 'description' in prompt. 
        # We can accept dict in mix_and_create or convert dict to string there.
        # Let's return the object (Pydantic model) and convert to string where needed.
        return response

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
            visual_prompt: str = Field(description="Full prompt for AI image generator")
            # Structured breakdown
            subject: str = Field(description="The main character/subject")
            action: str = Field(description="What the subject is doing")
            context: str = Field(description="Environment or background elements")
            art_style: str = Field(description="The specific art style used")
            colors: str = Field(description="The color palette used")
            
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Business logic or why this design works")
            focus: str = Field(description="The specific element that was changed, e.g., 'Subject', 'Action', 'Style', 'Context'")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

        prompt = f"""
        You are a Creative Director for a POD (Print on Demand) T-shirt business.
        
        Original Image Analysis:
        {description}
        
        Inspiration from our Database (Trending Keywords/Styles):
        {rag_ideas}
        
        Task:
        Create 4 DISTINCT T-shirt design concepts based on the following specific strategies:
        
        1. **Concept 1: The Subject Twist**
           - **CRITICAL**: Look at the "Action" identified in the image analysis.
           - **KEEP THIS ACTION** exactly as is.
           - **CHANGE THE SUBJECT** to a completely different character based on RAG keywords (e.g. Instead of Bigfoot holding a sausage, make it a T-Rex holding a sausage).
           
        2. **Concept 2: The Action Switch**
           - **CRITICAL**: Keep the **ORIGINAL SUBJECT**.
           - **CHANGE THE ACTION** to something unexpected, funny, or trending (e.g. skateboarding, gaming, drinking boba).
           
        3. **Concept 3: The Visual Remix (Style + Color)**
           - Keep the original subject and action.
           - **CHANGE THE ART STYLE AND COLORS** together. Use a trending style (e.g. Kawaii, Vaporwave, Glitch Art) AND a matching unique color palette.
           
        4. **Concept 4: The Context Shift**
           - Keep the original subject and action.
           - **CHANGE THE CONTEXT/BACKGROUND** to a completely different setting (e.g. Outer Space, Underwater, Cyberpunk City, Ancient Rome).
        """
        
        if user_instruction:
            prompt += f"""
            
            IMMEDIATE USER INSTRUCTION:
            "{user_instruction}"
            
            CRITICAL OVERRIDE: 
            If the user instruction starts with "Refining concept...", it means they want to IMPROVE a specific concept they already saw.
            In this case:
            1. **FOCUS ONLY ON THAT CONCEPT'S CORE IDEA**.
            2. **APPLY THE USER'S REQUESTED CHANGE AS THE NEW TRUTH**.
               - If the user asks to "change action to dancing", the new Action IS "dancing". **DO NOT** keep the old action.
               - If the user asks to "change subject to a cat", the new Subject IS "cat". **DO NOT** keep the old subject.
            3. Generate 3 VARIATIONS of this *refined* concept (e.g. 3 different ways to show the T-Rex dancing).
            4. You can ignore the strict "Subject/Style/Color" split if it doesn't make sense for a refinement, BUT try to keep diversity in the details.
            """
            
        prompt += """
        
        Output Format:
        Return a JSON with 4 concepts.
        For each concept, provide:
        - title: Catchy name
        - visual_prompt: The full, detailed prompt for generation.
        - subject: The specific subject (e.g. "T-Rex").
        - action: The specific action (e.g. "holding a sausage").
        - context: Background/Environment details.
        - art_style: The art style used.
        - colors: The color palette.
        - caption: Text for the shirt.
        - logic: Explain WHICH strategy you used (Subject Twist, Style Shift, or Color Pop) and why.
        - focus: ONE word indicating the main change: "Subject", "Action", "Style", or "Context".
        """
        
        structured_llm = self.creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.concepts

    def run(self, image_path, user_instruction=None):
        # 1. Vision Analysis (Structured)
        vision_analysis = self.analyze_image(image_path)
        
        # Convert to string for downstream tasks
        if hasattr(vision_analysis, 'model_dump_json'):
             description_str = vision_analysis.model_dump_json()
             analysis_dict = vision_analysis.model_dump()
        else:
             description_str = str(vision_analysis)
             analysis_dict = {"raw_text": description_str} # Fallback
             
        # 2. RAG Retrieval
        rag_ideas = self.retrieve_ideas(vision_analysis, user_instruction)
        
        # 3. Creative Generation
        concepts = self.mix_and_create(description_str, rag_ideas, user_instruction)
        
        return {
            "analysis": description_str, # Keep string for backward compat if needed, or update frontend to use 'vision_analysis'
            "vision_analysis": analysis_dict, # Pass structured data
            "rag_context": rag_ideas,
            "concepts": [c.model_dump() for c in concepts]
        }

    def remix_concept(self, original_concept: dict):
        print(f"Remixing concept: {original_concept.get('title')}...")
        from pydantic import BaseModel, Field
        from typing import List

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Detailed prompt for AI image generator")
            subject: str = Field(description="The specific subject")
            action: str = Field(description="The specific action")
            context: str = Field(description="Environment or background")
            art_style: str = Field(description="The art style")
            colors: str = Field(description="The color palette")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Why this variation works")
            focus: str = Field(description="The specific element that was changed")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

        focus_area = original_concept.get('focus', 'General')
        
        # tailoring the prompt based on what kind of concept this is
        specific_instruction = ""
        if focus_area == 'Subject':
            specific_instruction = "This concept was about a Subject Twist. Create 1 NEW VARIATION with a **DIFFERENT SUBJECT** performing the SAME Action. **CRITICAL: You MUST KEEP the original Action, Context, Art Style, and Colors EXACTLY AS IS.**"
        elif focus_area == 'Action':
            specific_instruction = "This concept was about an Action Switch. Create 1 NEW VARIATION with the SAME Subject performing a **DIFFERENT ACTION**. **CRITICAL: You MUST KEEP the original Subject, Context, Art Style, and Colors EXACTLY AS IS.**"
        elif focus_area == 'Style' or focus_area == 'Visual':
            specific_instruction = "This concept was about a Visual Remix. Create 1 NEW VARIATION with the SAME Subject/Action/Context but a **DIFFERENT ART STYLE AND COLOR PALETTE**. **CRITICAL: You MUST KEEP the original Subject, Action, and Context EXACTLY AS IS.**"
        elif focus_area == 'Context':
            specific_instruction = "This concept was about a Context Shift. Create 1 NEW VARIATION with the SAME Subject/Action but in a **DIFFERENT CONTEXT**. **CRITICAL: You MUST KEEP the original Subject, Action, Art Style, and Colors EXACTLY AS IS.**"
        else:
             specific_instruction = "Create 1 BOLD variation of this concept using SCAMPER techniques."

        prompt = f"""
        You are a Creative Director for a POD T-shirt business.
        
        Original Concept:
        Title: {original_concept.get('title')}
        Visual: {original_concept.get('visual_prompt')}
        Caption: {original_concept.get('caption')}
        Subject: {original_concept.get('subject')}
        Action: {original_concept.get('action')}
        Context: {original_concept.get('context')}
        Art Style: {original_concept.get('art_style')}
        Colors: {original_concept.get('colors')}
        Focus: {focus_area}
        
        Task:
        {specific_instruction}
        
        CRITICAL RULES:
        1. **Generate ONLY 1 Concept**.
        2. **STRICTLY Maintain Non-Focus Elements**: If the focus is Context, DO NOT change the Art Style or Subject. If the focus is Action, DO NOT change the Context.
        3. **New Caption**: Update the caption to match the new variation.
        4. **Detailed Visual Prompt**: Write a full prompt for image generation that explicitly describes the elements to keep.
        """
        
        structured_llm = self.creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return [c.model_dump() for c in response.concepts]

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
