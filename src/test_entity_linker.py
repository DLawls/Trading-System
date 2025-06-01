"""
Test script for EntityLinker functionality
"""

from src.event_detection.entity_linker import EntityLinker, EntityType

def test_entity_linker():
    """Test the EntityLinker with various news texts."""
    
    print("🔗 Testing EntityLinker...")
    
    # Initialize EntityLinker
    linker = EntityLinker()
    
    # Test texts with different entity types
    test_texts = [
        {
            'title': 'Tesla Earnings Beat',
            'text': 'Tesla Inc. (TSLA) reported strong Q4 earnings that beat analyst estimates. CEO Elon Musk highlighted record production numbers.',
            'description': 'Simple earnings news with ticker and person'
        },
        
        {
            'title': 'Microsoft OpenAI Deal',
            'text': 'Microsoft Corporation announced a $10 billion investment in OpenAI, strengthening its position in artificial intelligence. The deal values OpenAI at $29 billion.',
            'description': 'M&A news with financial amounts'
        },
        
        {
            'title': 'Apple Product Launch',
            'text': 'Apple unveiled the new iPhone 15 with advanced AI features. The device will be manufactured in Cupertino and shipped worldwide.',
            'description': 'Product news with location'
        },
        
        {
            'title': 'Federal Reserve Decision',
            'text': 'The Federal Reserve raised interest rates by 0.75% to combat inflation. Chairman Jerome Powell stated this is necessary for economic stability.',
            'description': 'Economic news with person and organization'
        },
        
        {
            'title': 'Amazon Acquisition',
            'text': 'Amazon.com Inc. (AMZN) is acquiring Whole Foods for $13.7 billion. Jeff Bezos called it a strategic move into physical retail.',
            'description': 'Acquisition with financial details'
        },
        
        {
            'title': 'Netflix Partnership',
            'text': 'Netflix, Inc. announced a content partnership with Disney and Warner Bros. The streaming giant will expand its catalog globally.',
            'description': 'Partnership between multiple companies'
        },
        
        {
            'title': 'Crypto Market Update',
            'text': 'Bitcoin (BTC-USD) surged past $50,000 while Ethereum (ETH-USD) reached new highs. Coinbase reported record trading volumes.',
            'description': 'Cryptocurrency with multiple tickers'
        },
        
        {
            'title': 'Bank Merger News',
            'text': 'JPMorgan Chase & Co. (JPM) and Bank of America Corp. are reportedly in merger discussions. The combined entity would create the largest US bank.',
            'description': 'Financial sector merger'
        }
    ]
    
    print(f"\n📰 Analyzing {len(test_texts)} news texts for entities...\n")
    
    all_entities = []
    
    for i, test_case in enumerate(test_texts, 1):
        print(f"🔍 Example {i}: {test_case['title']}")
        print(f"   📄 Description: {test_case['description']}")
        
        # Combine title and text for entity extraction
        full_text = f"{test_case['title']} {test_case['text']}"
        
        # Extract entities
        entities = linker.link_entities(full_text)
        all_entities.extend(entities)
        
        print(f"   🎯 Found {len(entities)} entities:")
        
        if entities:
            # Group entities by type
            entity_groups = {}
            for entity in entities:
                entity_type = entity.entity_type.value
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                entity_groups[entity_type].append(entity)
            
            # Display by type
            for entity_type, group in entity_groups.items():
                print(f"     {entity_type.upper()}:")
                for entity in group:
                    ticker_info = f" ({entity.ticker})" if entity.ticker else ""
                    confidence_info = f" [{entity.confidence:.2f}]"
                    print(f"       • {entity.text}{ticker_info}{confidence_info}")
                    if entity.canonical_name and entity.canonical_name != entity.text:
                        print(f"         → {entity.canonical_name}")
                    if entity.aliases:
                        print(f"         Sector: {', '.join(entity.aliases)}")
        else:
            print("     ❌ No entities detected")
        
        print("-" * 80)
    
    # Summary analysis
    print("\n📋 ENTITY EXTRACTION SUMMARY")
    print("=" * 50)
    
    if all_entities:
        # Count by entity type
        type_counts = {}
        financial_entities = linker.get_financial_entities(all_entities)
        
        for entity in all_entities:
            entity_type = entity.entity_type.value
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        print("📊 Entity Type Distribution:")
        for entity_type, count in sorted(type_counts.items()):
            print(f"   {entity_type.capitalize()}: {count} entities")
        
        print(f"\n💰 Financial Entities: {len(financial_entities)} / {len(all_entities)}")
        
        # Show top financial entities
        if financial_entities:
            print(f"\n🏢 Top Financial Entities Found:")
            
            # Sort by confidence and show unique tickers/companies
            unique_financial = {}
            for entity in financial_entities:
                key = entity.ticker or entity.canonical_name or entity.text
                if key not in unique_financial or entity.confidence > unique_financial[key].confidence:
                    unique_financial[key] = entity
            
            sorted_financial = sorted(unique_financial.values(), key=lambda x: x.confidence, reverse=True)
            
            for i, entity in enumerate(sorted_financial[:10], 1):  # Top 10
                ticker_info = f" ({entity.ticker})" if entity.ticker else ""
                print(f"   {i}. {entity.canonical_name or entity.text}{ticker_info} [{entity.confidence:.2f}]")
        
        # Sector analysis
        sectors_found = set()
        for entity in all_entities:
            if entity.aliases:
                for alias in entity.aliases:
                    if alias.startswith('sector:'):
                        sectors_found.add(alias.replace('sector:', ''))
        
        if sectors_found:
            print(f"\n🏭 Sectors Identified: {', '.join(sorted(sectors_found))}")
        
        # Convert to DataFrame for analysis
        df = linker.to_dataframe(all_entities)
        print(f"\n📋 Entity Analysis DataFrame:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if not df.empty:
            # Show companies with tickers
            companies_with_tickers = df[df['ticker'].notna()]
            if not companies_with_tickers.empty:
                print(f"\n📈 Companies with Stock Tickers ({len(companies_with_tickers)}):")
                for _, row in companies_with_tickers.iterrows():
                    print(f"   • {row['canonical_name']} ({row['ticker']}) - {row['confidence']:.2f}")
        
        # Test specific functionality
        print(f"\n🧪 Testing Additional Functionality:")
        
        # Test with a complex financial text
        complex_text = """
        Apple Inc. (AAPL) and Microsoft Corporation (MSFT) are the largest tech companies by market cap.
        Tesla's CEO Elon Musk tweeted about Bitcoin, causing BTC-USD to surge.
        JPMorgan Chase analysts upgraded Amazon stock to buy.
        """
        
        complex_entities = linker.link_entities(complex_text)
        financial_only = linker.get_financial_entities(complex_entities)
        
        print(f"   Complex text entities: {len(complex_entities)}")
        print(f"   Financial entities only: {len(financial_only)}")
        
        # Show financial entities from complex text
        for entity in financial_only:
            print(f"     • {entity.text} ({entity.entity_type.value}) - {entity.ticker or 'N/A'}")
    
    print(f"\n✅ EntityLinker testing complete!")

if __name__ == "__main__":
    test_entity_linker() 