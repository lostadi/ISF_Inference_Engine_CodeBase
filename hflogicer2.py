#!/usr/bin/env python3
"""
neuro_symbolic_llm.py
Enhanced neuro-symbolic framework: Runs locally with Ollama for generation (using phi4-reasoning-abliterated for advanced reasoning), handles file uploads (text/PDF), and produces detailed propositions/derivations.

Local Setup:
- Install: pip install ollama transformers torch streamlit networkx matplotlib pdfplumber
- Install Ollama: Follow https://ollama.com/ instructions, then pull the model: ollama pull huihui_ai/phi4-reasoning-abliterated
- (Optional alternatives: ollama pull huihui_ai/jan-nano-abliterated or others listed)
- Run: streamlit run neuro_symbolic_llm.py


Changes Made
I selected huihui_ai/phi4-reasoning-abliterated as the primary model for text generation in this update, as it aligns with your preference (it's based on Phi-3/4 variants optimized for reasoning tasks, which fits the neuro-symbolic framework's emphasis on logical inference). I defaulted to this in the code but kept flexibility for alternatives like jan-nano-abliterated:latest (you can pass it as an argument to NeuroSymbolicLLM(ollama_model='jan-nano-abliterated') if needed). The other options (acereason-nemotron-abliterated:7b and qwen3-abliterated:4b) are compatible via Ollama but not defaulted, as Phi-4 provides a strong balance of reasoning capability and efficiency for this use case.

Here's a specific breakdown of changes from the previous version (which used transformers for everything):

Overall Script and Documentation

Docstring: Updated to specify Ollama integration and the chosen model (huihui_ai/phi4-reasoning-abliterated). Added setup notes for Ollama installation and model pulling. Mentioned alternatives from your list for easy switching.
Imports:
Added ollama for local LLM generation.
Added import numpy as np explicitly (used for embedding mean pooling; was implicit before via 'import numpy as np' in functions).
No Changes to Symbolic Logic Framework or Inference Engine: These remained unchanged, as the focus was on switching the generation backend.
Symbolic Abstraction Engine

Initialization: Removed the generator parameter (previously a transformers pipeline). Now only takes embedder, as generation is handled globally via Ollama.
extract_entities_and_relations:
Switched from self.generator(prompt, ...)[0]['generated_text'] (transformers) to ollama.generate(model='huihui_ai/phi4-reasoning-abliterated', prompt=prompt)['response'].
This leverages the model's reasoning strengths for better extraction of entities, relations, and propositions.
Neuro-Symbolic Integration (NeuroSymbolicLLM)

Initialization:
Added ollama_model parameter with default 'huihui_ai/phi4-reasoning-abliterated'.
Included ollama.pull(self.ollama_model) to ensure the model is downloaded/available (with error handling).
Kept the embedder as transformers-based (all-MiniLM-L6-v2), as Ollama doesn't natively support feature extraction— this hybrid approach maintains efficiency.
process_input:
In the refinement loop, changed generation to ollama.generate(model=self.ollama_model, prompt=refinement_prompt)['response'].
_generate_response:
Switched to ollama.generate(model=self.ollama_model, prompt=prompt)['response'] for response generation, removing transformers-specific params like max_new_tokens/temperature (Ollama handles these internally or via options, but defaults suffice for reasoning tasks).
No Changes to Other Methods: Like add_rule, visualize_knowledge_graph, etc., as they don't involve generation.
Streamlit Application

Initialization Spinner: Updated message to "Initializing Ollama and embedder..." to reflect the new backend.
No Other Changes: The UI and file handling remain the same.
These changes reduce dependency on large transformers models for generation (Phi-4 via Ollama is lighter and reasoning-focused), improve setup simplicity with Ollama, and maintain local execution. The code is still compatible with your other model preferences— just change the default or pass the tag (e.g., 'jan-nano-abliterated:latest').

How to Setup and Use
Setup

Install Dependencies:
Run: pip install ollama transformers torch streamlit networkx matplotlib pdfplumber numpy
ollama: For the LLM backend.
transformers torch: For embeddings (kept hybrid for efficiency).
Others: For UI, graphs, PDF handling, etc.
Install Ollama:
Download and install from the official site: https://ollama.com/ (available for macOS, Linux, Windows).
Start the Ollama server: Run ollama serve in a terminal (or it starts automatically on some systems).
Pull the Model:
In a terminal: ollama pull huihui_ai/phi4-reasoning-abliterated
This downloads the model (~2-4GB depending on quantization).
Alternatives:
For your favorite: ollama pull jan-nano-abliterated:latest
Others: ollama pull acereason-nemotron-abliterated:7b or ollama pull qwen3-abliterated:4b (then update the default in code or pass as arg).
Note: If using alternatives, ensure they support reasoning prompts well; Phi-4 was chosen for its ablation optimized for logic tasks.
Hardware Requirements:
CPU/GPU: Works on CPU, but GPU (e.g., NVIDIA with CUDA) speeds up embeddings if torch detects it.
RAM: At least 8GB (model-dependent; Phi-4 is efficient).
Run the App:
In a terminal: streamlit run neuro_symbolic_llm.py
Open the browser URL provided (usually http://localhost:8501).
Usage

Basic Interaction:
In the Streamlit UI (left column: "Interaction"):
Upload a text (.txt) or PDF file: It extracts content (full text from PDFs via pdfplumber) and processes it symbolically.
Or enter text in the chat input: Appends to file content if uploaded.
Click enter/submit: Processes input through abstraction → inference → refinement → generation.
Output: Displays the reasoned response (structured with propositions, derivations, etc.). Expand "Symbolic Output" for formulas, facts, steps, and derivations.
Adding Rules/Constraints/Ontology:
Sidebar: Use forms to add logical rules (e.g., "If Person(x)" → "Mortal(x)"), constraints, or ontology relations (e.g., subclass "Dog" of "Animal").
These integrate into the KB for future inferences.
Viewing Results:
Right column: "KB Graph" visualizes entities/predicates/ontology.
"Recent Facts": Lists last 10 symbolic facts.
Clear KB: Button in sidebar resets everything.
Switching Models:
Edit code: Change default in NeuroSymbolicLLM init (e.g., ollama_model='jan-nano-abliterated:latest').
Or instantiate dynamically if customizing.
Tips:
For questions/goals: Include '?' in input to trigger backward chaining.
File Processing: Large PDFs may take time; content is truncated in prompts (~2000 chars) but fully abstracted.
Debugging: If Ollama errors (e.g., model not found), ensure server is running and model pulled.
Performance: On CPU, generation may be slow (~10-30s per call); GPU accelerates embeddings.
This setup ensures efficient, local reasoning with your preferred abliterated models! If issues arise, check Ollama logs or try alternatives.


"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
import ollama  # For local generation with Ollama
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import pdfplumber
import functools
import numpy as np  # For embedding mean pooling

# =============================================================================
# SYMBOLIC LOGIC FRAMEWORK (Unchanged)
# =============================================================================

class LogicalOperator(Enum):
    AND = "∧"
    OR = "∨" 
    NOT = "¬"
    IMPLIES = "→"
    EQUIV = "↔"
    FORALL = "∀"
    EXISTS = "∃"
    NECESSARY = "□"
    POSSIBLE = "◇"

@dataclass(eq=True, frozen=True)
class Var:
    name: str

def is_variable(term: Union[str, Var]) -> bool:
    return isinstance(term, Var)

@dataclass
class Predicate:
    name: str
    arity: int
    args: List[Union[str, Var]] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    
    def __str__(self):
        args_str = ', '.join(a.name if is_variable(a) else a for a in self.args)
        return f"{self.name}({args_str})"
    
    def __hash__(self):
        args_tuple = tuple(a.name if is_variable(a) else a for a in self.args)
        return hash((self.name, self.arity, args_tuple))

@dataclass 
class LogicalFormula:
    operator: Optional[LogicalOperator] = None
    predicates: List[Predicate] = field(default_factory=list)
    subformulas: List['LogicalFormula'] = field(default_factory=list)
    quantified_vars: List[str] = field(default_factory=list)
    
    def __str__(self):
        if self.operator is None and len(self.predicates) == 1:
            return str(self.predicates[0])
        
        if self.operator == LogicalOperator.NOT:
            return f"¬{self.subformulas[0]}"
        
        if self.operator in [LogicalOperator.AND, LogicalOperator.OR, LogicalOperator.IMPLIES]:
            op_str = self.operator.value
            if len(self.subformulas) >= 2:
                return f"({self.subformulas[0]} {op_str} {self.subformulas[1]})"
        
        if self.operator in [LogicalOperator.FORALL, LogicalOperator.EXISTS]:
            vars_str = ", ".join(self.quantified_vars)
            return f"{self.operator.value}{vars_str}. {self.subformulas[0]}"
        
        if self.operator in [LogicalOperator.NECESSARY, LogicalOperator.POSSIBLE]:
            return f"{self.operator.value}{self.subformulas[0]}"
            
        return str(self.predicates)

    def __hash__(self):
        return hash((self.operator, tuple(self.predicates), tuple(self.subformulas), tuple(self.quantified_vars)))

class KnowledgeBase:
    def __init__(self):
        self.facts: Set[LogicalFormula] = set()
        self.rules: List[Tuple[LogicalFormula, LogicalFormula]] = [] 
        self.constraints: List[LogicalFormula] = []
        self.entities: Set[str] = set()
        self.predicates: Dict[str, Predicate] = {}
        self.ontology_graph: nx.DiGraph = nx.DiGraph() 
    
    def add_fact(self, fact: LogicalFormula):
        self.facts.add(fact)
        self._extract_entities_and_predicates(fact)
    
    def add_rule(self, premise: LogicalFormula, conclusion: LogicalFormula):
        self.rules.append((premise, conclusion))
        self._extract_entities_and_predicates(premise)
        self._extract_entities_and_predicates(conclusion)
    
    def add_constraint(self, constraint: LogicalFormula):
        self.constraints.append(constraint)
        self._extract_entities_and_predicates(constraint)
    
    def add_ontology_relation(self, subclass: str, superclass: str):
        self.ontology_graph.add_edge(subclass, superclass, type='subclass_of')
    
    def _extract_entities_and_predicates(self, formula: LogicalFormula):
        for pred in formula.predicates:
            self.predicates[pred.name] = pred
            for arg in pred.args:
                if not is_variable(arg):
                    self.entities.add(arg)
        
        for sub in formula.subformulas:
            self._extract_entities_and_predicates(sub)
    
    def query_facts_by_predicate(self, predicate_name: str) -> List[LogicalFormula]:
        return [fact for fact in self.facts 
                if any(p.name == predicate_name for p in fact.predicates)]
    
    def get_applicable_rules(self, fact: LogicalFormula) -> List[Tuple[LogicalFormula, LogicalFormula]]:
        applicable = []
        for premise, conclusion in self.rules:
            if unify_formula(fact, premise) is not None:
                applicable.append((premise, conclusion))
        return applicable

@functools.lru_cache(maxsize=1024)
def unify_formula(f1: LogicalFormula, f2: LogicalFormula) -> Optional[Dict[Var, Union[str, Var]]]:
    if f1.operator != f2.operator:
        return None
    if len(f1.predicates) != len(f2.predicates):
        return None
    subst = {}
    for p1, p2 in zip(f1.predicates, f2.predicates):
        sub = unify(p1, p2, subst)
        if sub is None:
            return None
        subst = sub
    for s1, s2 in zip(f1.subformulas, f2.subformulas):
        sub = unify_formula(s1, s2)
        if sub is None:
            return None
        subst.update(sub)
    if len(f1.quantified_vars) != len(f2.quantified_vars):
        return None
    return subst

def unify(pred1: Predicate, pred2: Predicate, subst: Dict[Var, Union[str, Var]] = None) -> Optional[Dict[Var, Union[str, Var]]]:
    if subst is None:
        subst = {}
    if pred1.name != pred2.name or pred1.arity != pred2.arity:
        return None
    for a1, a2 in zip(pred1.args, pred2.args):
        subst = unify_terms(a1, a2, subst)
        if subst is None:
            return None
    return subst

def unify_terms(t1: Union[str, Var], t2: Union[str, Var], subst: Dict[Var, Union[str, Var]]) -> Optional[Dict[Var, Union[str, Var]]]:
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)
    if t1 == t2:
        return subst
    if is_variable(t1):
        if occur_check(t1, t2):
            return None
        subst[t1] = t2
        return subst
    if is_variable(t2):
        if occur_check(t2, t1):
            return None
        subst[t2] = t1
        return subst
    return None

def apply_subst(t: Union[str, Var], subst: Dict[Var, Union[str, Var]]) -> Union[str, Var]:
    if is_variable(t) and t in subst:
        return apply_subst(subst[t], subst)
    return t

def occur_check(var: Var, t: Union[str, Var]) -> bool:
    if var == t:
        return True
    return False

def apply_subst_to_formula(formula: LogicalFormula, subst: Dict[Var, Union[str, Var]]) -> LogicalFormula:
    new_predicates = [apply_subst_to_pred(p, subst) for p in formula.predicates]
    new_subformulas = [apply_subst_to_formula(f, subst) for f in formula.subformulas]
    return LogicalFormula(formula.operator, new_predicates, new_subformulas, formula.quantified_vars)

def apply_subst_to_pred(pred: Predicate, subst: Dict[Var, Union[str, Var]]) -> Predicate:
    new_args = [apply_subst(arg, subst) for arg in pred.args]
    return Predicate(pred.name, pred.arity, new_args, pred.embedding)

# =============================================================================
# SYMBOLIC ABSTRACTION ENGINE (Updated for Ollama generation)
# =============================================================================

class SymbolicAbstractionEngine:
    def __init__(self, embedder: pipeline):
        self.embedder = embedder
        self.entity_patterns = {
            'person': r'\b(?:person|people|individual|human|man|woman|student|teacher|doctor)\b',
            'location': r'\b(?:place|location|city|country|building|room|office)\b',
            'action': r'\b(?:runs?|walks?|teaches?|learns?|loves?|likes?|knows?)\b',
            'property': r'\b(?:tall|short|smart|kind|red|blue|large|small)\b'
        }
        
    def extract_entities_and_relations(self, text: str) -> Dict[str, List[str]]:
        prompt = f"""
        Extract entities, relationships, and properties from this text in structured JSON format.
        Focus on logical propositions and potential derivations.
        Text: "{text[:2000]}"
        
        JSON:
        {{
            "entities": ["entity1", "entity2"],
            "relations": [["relation", "entity1", "entity2"]],
            "properties": [["property", "entity"]],
            "quantifiers": ["all", "some", "none"],
            "propositions": ["prop1", "prop2"]
        }}
        """
        
        try:
            response = ollama.generate(model='huihui_ai/phi4-reasoning-abliterated', prompt=prompt)['response']
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return self._pattern_based_extraction(text)
    
    def _pattern_based_extraction(self, text: str) -> Dict[str, List[str]]:
        entities = []
        relations = []
        properties = []
        propositions = text.split('. ') 
        
        words = text.lower().split()
        for word in words:
            if re.search(self.entity_patterns['person'], word):
                entities.append(word)
            elif re.search(self.entity_patterns['action'], word):
                relations.append([word])
            elif re.search(self.entity_patterns['property'], word):
                properties.append([word])
        
        return {
            "entities": list(set(entities)),
            "relations": relations, 
            "properties": properties,
            "quantifiers": [],
            "propositions": propositions
        }
    
    def translate_to_predicate_logic(self, text: str) -> List[LogicalFormula]:
        extracted = self.extract_entities_and_relations(text)
        entities = extracted.get("entities", [])
        relations = extracted.get("relations", [])
        properties = extracted.get("properties", [])
        quantifiers = extracted.get("quantifiers", [])
        propositions = extracted.get("propositions", [])
        
        formulas = []
        
        for entity in entities:
            entity_type = self._infer_entity_type(entity, text)
            pred = Predicate(entity_type, 1, [entity])
            self._ground_symbol(pred, text)
            formulas.append(LogicalFormula(None, [pred]))
        
        for rel in relations:
            if len(rel) >= 3:
                pred = Predicate(rel[0], 2, rel[1:])
                self._ground_symbol(pred, text)
                formulas.append(LogicalFormula(None, [pred]))
        
        for prop in properties:
            if len(prop) >= 2:
                pred = Predicate(prop[0], 1, [prop[1]])
                self._ground_symbol(pred, text)
                formulas.append(LogicalFormula(None, [pred]))
        
        for prop in propositions:
            pred = Predicate("Proposition", 1, [prop[:50]]) 
            formulas.append(LogicalFormula(None, [pred]))
        
        if quantifiers and formulas:
            op = LogicalOperator.FORALL if "all" in quantifiers else LogicalOperator.EXISTS
            formulas[0] = LogicalFormula(op, [], [formulas[0]], ["x"])
        
        return formulas
    
    def _infer_entity_type(self, entity: str, context: str) -> str:
        context_lower = context.lower()
        if 'person' in context_lower or 'human' in context_lower:
            return "Person"
        if 'place' in context_lower or 'city' in context_lower:
            return "Location"
        return "Concept"
    
    def _ground_symbol(self, pred: Predicate, context: str):
        try:
            ground_text = f"{str(pred)} in context: {context[:200]}"
            emb_response = self.embedder(ground_text)[0]
            emb = np.mean(emb_response, axis=0).tolist()
            pred.embedding = emb
        except:
            pass

# =============================================================================
# SYMBOLIC INFERENCE ENGINE (Unchanged)
# =============================================================================

class SymbolicInferenceEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.inference_steps: List[Dict] = []
    
    def forward_chain(self, max_iterations: int = 10) -> List[LogicalFormula]:
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            facts_added = []
            
            for fact in list(self.kb.facts):
                for premise, conclusion in self.kb.get_applicable_rules(fact):
                    subst = unify_formula(fact, premise)
                    if subst:
                        derived = apply_subst_to_formula(conclusion, subst)
                        if derived not in self.kb.facts:
                            self.kb.add_fact(derived)
                            facts_added.append(derived)
                            new_facts.append(derived)
                            self.inference_steps.append({
                                'type': 'forward_chain',
                                'premise': str(fact),
                                'rule': f"{premise} → {conclusion}",
                                'conclusion': str(derived),
                                'iteration': iteration,
                                'derivation': f"Derived from {fact} using rule {premise} → {conclusion}"
                            })
            
            if not facts_added:
                break
                
            iteration += 1
        
        return new_facts
    
    def backward_chain(self, goal: LogicalFormula) -> bool:
        return self._prove_goal(goal, set())
    
    def _prove_goal(self, goal: LogicalFormula, visited: Set[str]) -> bool:
        goal_str = str(goal)
        if goal_str in visited:
            return False
        visited.add(goal_str)
        
        if any(unify_formula(goal, fact) for fact in self.kb.facts):
            return True
        
        for premise, conclusion in self.kb.rules:
            subst = unify_formula(goal, conclusion)
            if subst:
                sub_premise = apply_subst_to_formula(premise, subst)
                if self._prove_goal(sub_premise, visited.copy()):
                    self.inference_steps.append({
                        'type': 'backward_chain',
                        'goal': goal_str,
                        'rule': f"{premise} → {conclusion}",
                        'proved': True,
                        'derivation': f"Proved {goal} by reducing to {sub_premise}"
                    })
                    return True
        return False
    
    def check_consistency(self) -> Tuple[bool, List[str]]:
        inconsistencies = []
        
        for fact in self.kb.facts:
            neg_fact = LogicalFormula(LogicalOperator.NOT, [], [fact])
            if any(unify_formula(neg_fact, f) for f in self.kb.facts):
                inconsistencies.append(f"Direct contradiction: {fact} and ¬{fact}")
        
        for constraint in self.kb.constraints:
            if self.backward_chain(LogicalFormula(LogicalOperator.NOT, [], [constraint])):
                inconsistencies.append(f"Constraint violation: {constraint}")
        
        if nx.find_cycle(self.kb.ontology_graph, orientation='original'):
            inconsistencies.append("Ontology cycle detected")
        
        return not inconsistencies, inconsistencies

# =============================================================================
# NEURO-SYMBOLIC INTEGRATION (Updated to use Ollama)
# =============================================================================

class NeuroSymbolicLLM:
    def __init__(self, ollama_model='huihui_ai/phi4-reasoning-abliterated'):
        # Ollama for generation
        self.ollama_model = ollama_model
        # Check if model is available
        try:
            ollama.pull(self.ollama_model)
        except Exception as e:
            print(f"Warning: Could not pull {self.ollama_model}. Ensure Ollama is running. Error: {e}")
        
        # Embedder remains transformers-based
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
        embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embedder = pipeline("feature-extraction", model=embed_model, tokenizer=embed_tokenizer, device=0 if device == "cuda" else -1)
        
        self.knowledge_base = KnowledgeBase()
        self.abstraction_engine = SymbolicAbstractionEngine(self.embedder)
        self.inference_engine = SymbolicInferenceEngine(self.knowledge_base)
        self.conversation_history = []
        
    def process_input(self, input_text: str) -> Dict[str, Any]:
        logical_formulas = self.abstraction_engine.translate_to_predicate_logic(input_text)
        
        for formula in logical_formulas:
            self.knowledge_base.add_fact(formula)
        
        derived_facts = self.inference_engine.forward_chain()
        
        is_consistent, inconsistencies = self.inference_engine.check_consistency()
        
        refinement_rounds = 0
        max_refinements = 5
        while not is_consistent and refinement_rounds < max_refinements:
            error_msg = '; '.join(inconsistencies)
            refinement_prompt = f"""
            Error: {error_msg}
            Original: {input_text}
            Formulas: {', '.join(str(f) for f in logical_formulas)}
            
            Revise to resolve. Return JSON: {{"revised_text": "...", "revised_formulas": ["...", "..."]}}
            """
            response = ollama.generate(model=self.ollama_model, prompt=refinement_prompt)['response']
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                refined = json.loads(json_match.group())
                revised_text = refined.get('revised_text', input_text)
                revised_formulas = refined.get('revised_formulas', [])
                
                logical_formulas = []
                for f_str in revised_formulas:
                    pred_match = re.match(r'(\w+)\((.*)\)', f_str)
                    if pred_match:
                        name, args_str = pred_match.groups()
                        args = [Var(a.strip()) if a.strip().islower() else a.strip() for a in args_str.split(',')]
                        pred = Predicate(name, len(args), args)
                        logical_formulas.append(LogicalFormula(None, [pred]))
                
                self.knowledge_base.facts.clear()
                for formula in logical_formulas:
                    self.knowledge_base.add_fact(formula)
                
                derived_facts = self.inference_engine.forward_chain()
                is_consistent, inconsistencies = self.inference_engine.check_consistency()
                
                self.inference_engine.inference_steps.append({
                    'type': 'self_refinement',
                    'round': refinement_rounds,
                    'error': error_msg,
                    'revised': revised_text,
                    'fixed': is_consistent
                })
            refinement_rounds += 1
        
        proved = None
        goal = None
        if '?' in input_text:
            goal_formulas = self.abstraction_engine.translate_to_predicate_logic(input_text)
            if goal_formulas:
                goal = goal_formulas[0]
                proved = self.inference_engine.backward_chain(goal)
        
        response = self._generate_response(input_text, logical_formulas, derived_facts, inconsistencies, proved, goal)
        
        self.conversation_history.append({
            'input_text': input_text,
            'logical_formulas': [str(f) for f in logical_formulas],
            'derived_facts': [str(f) for f in derived_facts],
            'inconsistencies': inconsistencies,
            'proved': proved,
            'response': response
        })
        
        return {
            'response': response,
            'logical_formulas': logical_formulas,
            'derived_facts': derived_facts,
            'inconsistencies': inconsistencies,
            'proved': proved,
            'inference_steps': self.inference_engine.inference_steps[-10:],
            'kb_stats': self._get_kb_stats()
        }
    
    def _generate_response(self, input_text: str, formulas: List[LogicalFormula], 
                          derived_facts: List[LogicalFormula], inconsistencies: List[str],
                          proved: Optional[bool], goal: Optional[LogicalFormula]) -> str:
        symbolic_context = "\n".join([
            f"Propositions/Formulas: {', '.join(str(f) for f in formulas)}",
            f"Derivations/Facts: {', '.join(str(f) for f in derived_facts)}",
            f"Inconsistencies: {'; '.join(inconsistencies)}" if inconsistencies else "",
            f"Goal {str(goal)}: {'Proved' if proved else 'Not Proved'}" if goal else ""
        ])
        
        prompt = f"""
        Respond to: "{input_text}"
        
        Symbolic context (propositions and derivations):
        {symbolic_context}
        
        Structure:
        1. Key Propositions: List extracted logical statements.
        2. Derivations: Explain step-by-step inferences.
        3. Inconsistencies/Goals: Address any issues or proofs.
        4. Conclusion: Logically grounded answer.
        """
        
        try:
            gen_text = ollama.generate(model=self.ollama_model, prompt=prompt)['response']
            return gen_text.strip()
        except Exception as e:
            return f"Processing complete, context: {symbolic_context}. Error: {e}"
    
    def _get_kb_stats(self) -> Dict[str, int]:
        return {
            'facts': len(self.knowledge_base.facts),
            'rules': len(self.knowledge_base.rules),
            'constraints': len(self.knowledge_base.constraints),
            'entities': len(self.knowledge_base.entities),
            'predicates': len(self.knowledge_base.predicates),
            'ontology_relations': self.knowledge_base.ontology_graph.number_of_edges()
        }
    
    def add_rule(self, premise_text: str, conclusion_text: str):
        premise_formulas = self.abstraction_engine.translate_to_predicate_logic(premise_text)
        conclusion_formulas = self.abstraction_engine.translate_to_predicate_logic(conclusion_text)
        
        if premise_formulas and conclusion_formulas:
            self.knowledge_base.add_rule(premise_formulas[0], conclusion_formulas[0])
    
    def visualize_knowledge_graph(self) -> str:
        G = self.knowledge_base.ontology_graph.copy()
        
        for entity in self.knowledge_base.entities:
            G.add_node(entity, type='entity')
        
        for pred_name, pred in self.knowledge_base.predicates.items():
            args = [a.name if is_variable(a) else a for a in pred.args]
            if pred.arity == 2 and len(args) == 2:
                G.add_edge(args[0], args[1], label=pred_name, type='relation')
            else:
                G.add_node(str(pred), type='predicate')
                for arg in args:
                    G.add_edge(str(pred), arg, type='arg')
        
        pos = nx.spring_layout(G, k=1.2, iterations=50)
        
        plt.figure(figsize=(14, 10))
        
        node_colors = []
        for _, data in G.nodes(data=True):
            t = data.get('type')
            if t == 'entity':
                node_colors.append('lightblue')
            elif t == 'predicate':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightyellow')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
        
        edge_colors = ['blue' if G[u][v].get('type') == 'subclass_of' else 'gray' for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)
        
        nx.draw_networkx_labels(G, pos, font_size=9)
        
        edge_labels = {(u,v): d.get('label', d.get('type')) for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        
        plt.title('Knowledge Base Graph', fontsize=16)
        plt.axis('off')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=200, bbox_inches='tight')
            plt.close()
            return tmp.name

# =============================================================================
# STREAMLIT APPLICATION (Unchanged)
# =============================================================================

def main():
    st.set_page_config(page_title="Local Neuro-Symbolic LLM", layout="wide")
    
    st.title("🧠 Local Neuro-Symbolic LLM")
    st.caption("Runs locally with Ollama (phi4-reasoning-abliterated), handles files, detailed propositions/derivations")
    
    if 'nsllm' not in st.session_state:
        with st.spinner("Initializing Ollama and embedder..."):
            st.session_state.nsllm = NeuroSymbolicLLM()
    
    nsllm = st.session_state.nsllm
    
    with st.sidebar:
        st.header("📊 System State")
        kb_stats = nsllm._get_kb_stats()
        st.metric("Facts", kb_stats['facts'])
        st.metric("Rules", kb_stats['rules']) 
        st.metric("Constraints", kb_stats['constraints'])
        st.metric("Entities", kb_stats['entities'])
        st.metric("Predicates", kb_stats['predicates'])
        st.metric("Ontology Relations", kb_stats['ontology_relations'])
        
        st.header("⚙️ Add Rules")
        with st.form("add_rule"):
            premise = st.text_input("If (premise):")
            conclusion = st.text_input("Then (conclusion):")
            if st.form_submit_button("Add Rule"):
                nsllm.add_rule(premise, conclusion)
                st.success("Rule added!")
                st.rerun()
        
        st.header("🛡️ Add Constraints")
        constraint_text = st.text_input("Constraint:")
        if st.button("Add Constraint"):
            formulas = nsllm.abstraction_engine.translate_to_predicate_logic(constraint_text)
            if formulas:
                nsllm.knowledge_base.add_constraint(formulas[0])
                st.success("Added!")
                st.rerun()
        
        st.header("📚 Add Ontology Relation")
        subclass = st.text_input("Subclass:")
        superclass = st.text_input("Superclass:")
        if st.button("Add Relation"):
            nsllm.knowledge_base.add_ontology_relation(subclass, superclass)
            st.success("Added!")
            st.rerun()
        
        if st.button("🧹 Clear KB"):
            st.session_state.nsllm = NeuroSymbolicLLM()
            st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Interaction")
        
        uploaded_file = st.file_uploader("Upload Text/PDF File", type=['txt', 'pdf'])
        input_text = ""
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    input_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            else:
                input_text = uploaded_file.read().decode("utf-8")
            st.write("File content loaded (snippet):", input_text[:200] + "...")
        
        user_input = st.chat_input("Or enter text...")
        if user_input:
            input_text = user_input if not input_text else input_text + "\n" + user_input
        
        if input_text:
            with st.spinner("Processing..."):
                result = nsllm.process_input(input_text)
            
            st.write(result['response'])
            
            with st.expander("🔍 Symbolic Output"):
                st.write("**Propositions/Formulas:**")
                for f in result['logical_formulas']:
                    st.code(str(f))
                st.write("**Derivations/Facts:**")
                for f in result['derived_facts']:
                    st.code(str(f))
                if result.get('proved') is not None:
                    st.write(f"**Goal Provable:** {result['proved']}")
                if result['inconsistencies']:
                    st.write("**Inconsistencies:**")
                    for inc in result['inconsistencies']:
                        st.error(inc)
                if result['inference_steps']:
                    st.write("**Steps:**")
                    for step in result['inference_steps']:
                        st.json(step)
                        if 'derivation' in step:
                            st.write(step['derivation'])
    
    with col2:
        st.header("🕸️ KB Graph")
        if nsllm.knowledge_base.entities or nsllm.knowledge_base.ontology_graph.nodes:
            graph_file = nsllm.visualize_knowledge_graph()
            st.image(graph_file)
        else:
            st.info("Interact to build KB!")
        
        st.header("📋 Recent Facts")
        recent = list(nsllm.knowledge_base.facts)[-10:]
        if recent:
            for fact in recent:
                st.code(str(fact))
        else:
            st.info("No facts yet.")

if __name__ == "__main__":
    main()