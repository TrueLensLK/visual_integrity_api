import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from Universal_Detector.src.layers import forensic_case_builder
from Universal_Detector.src.layers import llm_judge
from Universal_Detector.src.layers import layer_5_judge
from Universal_Detector.src.layers import vision_agent

# Load environment variables (API Keys)
load_dotenv()

# 1. DEFINE THE GRAPH STATE
class ForensicState(TypedDict):
    image_path: str
    layer_scores: Dict[str, float]
    layer_details: Dict[str, str]
    vision_report: Dict[str, Any]
    rule_verdict: str
    rule_score: int
    rule_desc: str
    compiled_case: Dict[str, Any]
    final_verdict: str
    final_score: int
    final_desc: str
    llm_verdict_data: Any

# 2. DEFINE THE NODES

def run_tools_node(state: ForensicState):
    print(f"\n[Node: Tools] Analyzing {state['image_path']}...")
    # Simulate math layer outputs
    state["layer_scores"] = {
        "prnu": -45.0,
        "neural_network": 25.0,
        "face_consistency": -10.0
    }
    state["layer_details"] = {
        "prnu": "Unnatural noise grid detected.",
        "neural_network": "Looks like a natural photo.",
        "face_consistency": "Faces: 1 | Minor pupil asymmetry."
    }
    # Run Vision Agent
    try:
        state["vision_report"] = vision_agent.run_vision_agent(state["image_path"])
    except Exception as e:
        print(f"[Warning] Vision Agent failed: {e}")
        state["vision_report"] = {"visual_score": 0, "confidence": 0.5, "visual_uncertain": True}
    return state

def layer_5_node(state: ForensicState):
    print("[Node: Layer 5] Calculating baseline rule scores...")
    score, verdict, desc, effective_scores = layer_5_judge.calculate_integrity(
        prnu_score=state["layer_scores"].get("prnu", 0),
        visual_score=state["layer_scores"].get("neural_network", 0),
        # vlm_visual_score removed - not in signature
        visual_confidence=state["vision_report"].get("confidence", 1.0),
        visual_uncertain=state["vision_report"].get("visual_uncertain", False),
        c2pa_res={}, face_score=0, eye_score=0, meta_score=0, physics_score=0,
        watermark_score=0, watermark_desc="", context_score=0, context_details={},
        spectrum_score=0 # Added missing required argument
    )
    state["rule_score"] = score
    state["rule_verdict"] = verdict
    state["rule_desc"] = desc
    state["effective_scores"] = effective_scores
    return state

def case_builder_node(state: ForensicState):
    print("[Node: Case Builder] Compiling evidence into structured Case File...")
    state["compiled_case"] = forensic_case_builder.compile_case_file(
        image_path=state["image_path"],
        layer_scores=state["layer_scores"],
        layer_details=state["layer_details"],
        rule_based_verdict=state["rule_verdict"],
        rule_based_score=state["rule_score"],
        rule_based_description=state["rule_desc"],
        visual_confidence=state["vision_report"].get("confidence", 1.0),
        effective_scores=state.get("effective_scores", {})
    )
    return state

def final_judge_node(state: ForensicState):
    print("[Node: Final Judge] Evaluating contradictions...")
    judge = llm_judge.create_judge(enable_llm=True)
    final_verdict, final_score, final_desc, llm_data = judge.judge(
        case_file=state["compiled_case"],
        rule_based_verdict=state["rule_verdict"],
        rule_based_score=state["rule_score"],
        rule_based_description=state["rule_desc"],
        image_path=state["image_path"]
    )
    state["final_verdict"] = final_verdict
    state["final_score"] = final_score
    state["final_desc"] = final_desc
    state["llm_verdict_data"] = llm_data
    return state

# 3. WIRE THE GRAPH TOGETHER
workflow = StateGraph(ForensicState)
workflow.add_node("Run_Tools", run_tools_node)
workflow.add_node("Rule_Judge", layer_5_node)
workflow.add_node("Case_Builder", case_builder_node)
workflow.add_node("Final_Judge", final_judge_node)
workflow.set_entry_point("Run_Tools")
workflow.add_edge("Run_Tools", "Rule_Judge")
workflow.add_edge("Rule_Judge", "Case_Builder")
workflow.add_edge("Case_Builder", "Final_Judge")
workflow.add_edge("Final_Judge", END)
forensic_app = workflow.compile()

# RUNNER SCRIPT
if __name__ == "__main__":
    test_image = "test_image.jpg"
    print(f"=== STARTING FORENSIC PIPELINE FOR: {test_image} ===")
    initial_state = {
        "image_path": test_image,
        "layer_scores": {}, "layer_details": {}, "vision_report": {}
    }
    final_state = forensic_app.invoke(initial_state)
    print("\n=== FINAL PIPELINE VERDICT ===")
    print(f"Verdict: {final_state['final_verdict']}")
    print(f"Score:   {final_state['final_score']}/100")
    print(f"Details: {final_state['final_desc']}")
    if final_state['llm_verdict_data']:
        print(f"LLM Used: {final_state['llm_verdict_data'].source}")
