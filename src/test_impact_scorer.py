"""
Test script for ImpactScorer functionality
"""

from src.event_detection.event_classifier import EventClassifier, EventType
from src.event_detection.impact_scorer import ImpactScorer, ImpactLevel

def test_impact_scorer():
    """Test the ImpactScorer with various event types."""
    
    print("🎯 Testing ImpactScorer...")
    
    # Initialize components
    classifier = EventClassifier()
    scorer = ImpactScorer()
    
    # Test news headlines with different impact levels
    test_headlines = [
        ("Tesla reports record Q4 earnings, beats revenue estimates by 15%", 
         "Tesla Inc. announced quarterly results showing record revenue that beat analyst estimates..."),
        
        ("Apple announces new iPhone with breakthrough battery technology", 
         "Apple unveiled the new iPhone featuring revolutionary battery technology that extends life by 300%..."),
        
        ("Microsoft acquires OpenAI for $50 billion in landmark AI deal", 
         "Microsoft Corp. announced the acquisition of OpenAI in a strategic move to dominate AI..."),
        
        ("FDA approves Moderna's new cancer treatment drug", 
         "The Food and Drug Administration granted approval for Moderna's breakthrough cancer therapy..."),
        
        ("Tesla CEO Elon Musk steps down, new leadership appointed", 
         "Tesla announced that CEO Elon Musk will step down from his role..."),
        
        ("Amazon announces partnership with Walmart for logistics", 
         "Amazon and Walmart announced a strategic partnership to improve supply chain efficiency..."),
        
        ("Netflix stock price target raised by 3 analysts", 
         "Multiple Wall Street analysts upgraded Netflix with higher price targets..."),
        
        ("Federal Reserve raises interest rates by 0.75%", 
         "The Federal Reserve announced a significant interest rate increase in response to inflation...")
    ]
    
    print(f"\n📰 Analyzing {len(test_headlines)} news headlines for impact...\n")
    
    all_impact_scores = []
    
    for i, (title, content) in enumerate(test_headlines, 1):
        print(f"🔍 Example {i}: {title[:60]}...")
        
        # Classify the event
        events = classifier.classify_text(title, content)
        
        if events:
            # Take the highest confidence event
            primary_event = events[0]
            print(f"   📊 Event Type: {primary_event.event_type.value}")
            print(f"   🎯 Classification Confidence: {primary_event.confidence:.2f}")
            
            # Score the impact
            market_conditions = {
                'high_volatility': True,  # Simulate volatile market
                'earnings_season': 'earnings' in title.lower()
            }
            
            impact_score = scorer.score_event_impact(primary_event, market_conditions)
            all_impact_scores.append(impact_score)
            
            # Display results
            print(f"   💥 Impact Level: {impact_score.impact_level.value.upper()}")
            print(f"   📈 Impact Score: {impact_score.impact_score:.2f}")
            print(f"   💰 Expected Price Change: ±{impact_score.price_target_change:.1f}%")
            print(f"   📊 Volatility Increase: +{impact_score.volatility_increase:.1f}%")
            print(f"   🎯 Assessment Confidence: {impact_score.confidence:.2f}")
            
            # Show reasoning
            print(f"   🧠 Reasoning:")
            for reason in impact_score.reasoning[:3]:  # Show first 3 reasons
                print(f"      • {reason}")
            
        else:
            print("   ❌ No events detected")
        
        print("-" * 80)
    
    # Summary analysis
    print("\n📋 IMPACT ANALYSIS SUMMARY")
    print("=" * 50)
    
    if all_impact_scores:
        # Count by impact level
        impact_counts = {}
        for score in all_impact_scores:
            level = score.impact_level.value
            impact_counts[level] = impact_counts.get(level, 0) + 1
        
        print("📊 Impact Level Distribution:")
        for level, count in sorted(impact_counts.items()):
            print(f"   {level.capitalize()}: {count} events")
        
        # Highest impact events
        high_impact = scorer.get_high_impact_events(all_impact_scores, threshold=0.6)
        print(f"\n🚨 High Impact Events (score ≥ 0.6): {len(high_impact)}")
        
        for score in high_impact:
            print(f"   • {score.event.entity}: {score.event.title[:50]}...")
            print(f"     Impact: {score.impact_level.value} ({score.impact_score:.2f})")
        
        # Average impact by event type
        print(f"\n📈 Average Impact by Event Type:")
        event_type_impacts = {}
        for score in all_impact_scores:
            event_type = score.event.event_type.value
            if event_type not in event_type_impacts:
                event_type_impacts[event_type] = []
            event_type_impacts[event_type].append(score.impact_score)
        
        for event_type, scores in event_type_impacts.items():
            avg_impact = sum(scores) / len(scores)
            print(f"   {event_type}: {avg_impact:.2f}")
        
        # Convert to DataFrame for analysis
        df = scorer.to_dataframe(all_impact_scores)
        print(f"\n📋 Impact Analysis DataFrame:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if not df.empty:
            print(f"\n📊 Top 3 Highest Impact Events:")
            top_events = df.nlargest(3, 'impact_score')[['entity', 'event_type', 'impact_level', 'impact_score', 'price_target_change']]
            for idx, row in top_events.iterrows():
                print(f"   {idx+1}. {row['entity']} - {row['event_type']} ({row['impact_score']:.2f})")
    
    print(f"\n✅ ImpactScorer testing complete!")

if __name__ == "__main__":
    test_impact_scorer() 