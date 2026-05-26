import json
from QdrantRetriever import QdrantRetriever
from filters.BusinessLogicFilter import BusinessLogicFilter
from filters.DomainEventsFilter import DomainEventsFilter
from rag_llm_module.NL2IdentifyBusinessLogic import NL2IdentifyBusinessLogic
from rag_llm_module.NL2IdentifyDomainEvents import NL2IdentifyDomainEvents

def load_use_case(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

if __name__ == "__main__":
    # 1. start (Retriever)
    db_retriever = QdrantRetriever()

    # 2. equip (Filters)
    logic_filter = BusinessLogicFilter(db_retriever)
    events_filter = DomainEventsFilter(db_retriever)
    # ==== user input ====
    #raw_user_input = "When a customer places a food order, the system must process the payment successfully. After that, the kitchen is notified to prepare the meal."
    raw_user_input = user_input = load_use_case("test_case1.txt")

    print("==================================================")
    print("Pipeline Step 1: NL2IdentifyBusinessLogic")
    print("==================================================")
    
    # Use a filter to retrieve all the data -> feed it to the NL2 service.
    logic_examples = logic_filter.get_clean_examples(raw_user_input)
    step1_service = NL2IdentifyBusinessLogic()
    logic_output = step1_service.execute(raw_user_input, top_k_examples=logic_examples)
    #clean_logic_str = logic_output.get("BusinessLogic", raw_user_input)
    business_logic_list = logic_output.get("BusinessLogic", [])
    clean_logic_str = ", ".join(business_logic_list)

    print(f"\nStep 1 final output JSON:\n{json.dumps(logic_output, indent=2, ensure_ascii=False)}")
    print(f"Clean string passed to the next step: \"{clean_logic_str}\"")

    print("\n==================================================")
    print("Pipeline Step 2: NL2IdentifyDomainEvents")
    print("==================================================")
    
    # Use a filter to retrieve all the data -> feed it to the NL2 service.
    events_examples = events_filter.get_clean_examples(clean_logic_str)
    step2_service = NL2IdentifyDomainEvents()
    events_output = step2_service.execute(clean_logic_str, top_k_examples=events_examples)
    
    print(f"\nStep 2 final output JSON:\n{json.dumps(events_output, indent=2, ensure_ascii=False)}")