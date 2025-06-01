"""
Simple EntityLinker test without Unicode characters
"""

from event_detection.entity_linker import EntityLinker, EntityType

def test_entity_linker_simple():
    """Test EntityLinker with simple text examples"""
    
    print("Testing EntityLinker...")
    
    # Initialize EntityLinker
    linker = EntityLinker()
    
    # Test with simple news text
    test_text = "Tesla Inc. (TSLA) reported strong Q4 earnings that beat analyst estimates. CEO Elon Musk highlighted record production numbers."
    
    print(f"\nAnalyzing text: {test_text}")
    
    # Extract entities
    entities = linker.link_entities(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    
    for entity in entities:
        ticker_info = f" ({entity.ticker})" if entity.ticker else ""
        confidence_info = f" [{entity.confidence:.2f}]"
        print(f"  - {entity.entity_type.value}: {entity.text}{ticker_info}{confidence_info}")
        
        if entity.canonical_name and entity.canonical_name != entity.text:
            print(f"    -> Canonical: {entity.canonical_name}")
    
    # Test financial entity filtering
    financial_entities = linker.get_financial_entities(entities)
    print(f"\nFinancial entities: {len(financial_entities)}")
    
    for entity in financial_entities:
        print(f"  - {entity.text} ({entity.entity_type.value})")
    
    # Test with multiple companies
    multi_text = "Apple Inc. (AAPL) and Microsoft Corporation (MSFT) are the largest tech companies. Amazon stock is also performing well."
    
    print(f"\nAnalyzing multiple companies: {multi_text}")
    
    multi_entities = linker.link_entities(multi_text)
    financial_multi = linker.get_financial_entities(multi_entities)
    
    print(f"Found {len(financial_multi)} financial entities:")
    for entity in financial_multi:
        ticker_info = f" ({entity.ticker})" if entity.ticker else ""
        print(f"  - {entity.text}{ticker_info}")
    
    print("\nEntityLinker test completed successfully!")
    return True

if __name__ == "__main__":
    test_entity_linker_simple() 