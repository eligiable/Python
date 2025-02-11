import openai
from transformers import pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import random
from datetime import datetime, timedelta
import itertools
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd

class ABTest:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.tests: Dict[str, Dict] = {}

    def create_test(self, test_id: str, variants: List[Dict], metric: str):
        """Create a new A/B test."""
        self.tests[test_id] = {
            'variants': variants,
            'metric': metric,
            'results': {variant['id']: [] for variant in variants},
            'start_time': datetime.now(),
            'status': 'running'
        }
        return test_id

    def record_result(self, test_id: str, variant_id: str, metric_value: float):
        """Record a result for a specific variant."""
        if test_id in self.tests:
            self.tests[test_id]['results'][variant_id].append(metric_value)

    def analyze_test(self, test_id: str) -> Dict:
        """Analyze test results using statistical methods."""
        test = self.tests[test_id]
        results = []

        for variant in test['variants']:
            variant_id = variant['id']
            data = test['results'][variant_id]
            
            results.append({
                'variant_id': variant_id,
                'sample_size': len(data),
                'mean': np.mean(data) if data else 0,
                'std': np.std(data) if data else 0
            })

        # Perform statistical test (t-test)
        if len(results) >= 2:
            t_stat, p_value = stats.ttest_ind(
                test['results'][results[0]['variant_id']],
                test['results'][results[1]['variant_id']]
            )

            winner = max(results, key=lambda x: x['mean'])
            
            return {
                'test_id': test_id,
                'results': results,
                'p_value': p_value,
                'significant': p_value < (1 - self.confidence_level),
                'winner': winner['variant_id'] if p_value < (1 - self.confidence_level) else None
            }
        
        return {'error': 'Insufficient data'}

class ContentGenerator:
    def __init__(self, api_key: str):
        """Initialize content generator with API keys."""
        openai.api_key = api_key
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")

    async def generate_post(self, prompt: str, platform: str, style: str) -> str:
        """Generate social media post using AI."""
        system_prompt = f"""Create a {platform} post in a {style} style. 
        Follow these platform-specific guidelines:
        - Twitter: Max 280 characters, engaging hashtags
        - Instagram: Visual description, relevant hashtags
        - LinkedIn: Professional tone, industry insights
        - Facebook: Conversational, engaging"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

    def generate_variations(self, base_content: str, num_variations: int = 3) -> List[str]:
        """Generate variations of content for A/B testing."""
        variations = []
        
        for _ in range(num_variations):
            prompt = f"Rewrite this social media post in a different style while maintaining the core message: {base_content}"
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate a variation of this social media post."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            variations.append(response.choices[0].message.content)
        
        return variations

    def optimize_content(self, content: str, target_metrics: Dict[str, float]) -> str:
        """Optimize content based on target metrics."""
        current_metrics = self.analyze_content(content)
        
        if self._needs_optimization(current_metrics, target_metrics):
            prompt = f"""Optimize this social media post:
            Original: {content}
            Target metrics: {json.dumps(target_metrics)}
            Current metrics: {json.dumps(current_metrics)}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Optimize this social media post based on the metrics."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
        
        return content

    def _needs_optimization(self, current: Dict[str, float], target: Dict[str, float]) -> bool:
        """Check if content needs optimization based on metrics."""
        threshold = 0.1  # 10% difference threshold
        
        for metric, target_value in target.items():
            if metric in current:
                difference = abs(current[metric] - target_value) / target_value
                if difference > threshold:
                    return True
        
        return False

class CrossPlatformAnalytics:
    def __init__(self, analytics_data: List[Analytics]):
        """Initialize cross-platform analytics."""
        self.data = pd.DataFrame([vars(a) for a in analytics_data])
        self.metrics = ['engagement', 'likes', 'shares', 'comments', 'reach']

    def compare_platforms(self) -> Dict[str, Any]:
        """Compare performance across platforms."""
        platform_stats = self.data.groupby('platform')[self.metrics].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])

        # Normalize metrics for fair comparison
        normalized_metrics = {}
        for metric in self.metrics:
            normalized = self.data.pivot(columns='platform', values=metric)
            normalized = (normalized - normalized.mean()) / normalized.std()
            normalized_metrics[metric] = normalized.mean().to_dict()

        # Calculate platform effectiveness score
        effectiveness_scores = {}
        weights = {
            'engagement': 0.3,
            'reach': 0.3,
            'likes': 0.2,
            'shares': 0.1,
            'comments': 0.1
        }

        for platform in self.data['platform'].unique():
            score = sum(
                normalized_metrics[metric][platform] * weights[metric]
                for metric in self.metrics
                if platform in normalized_metrics[metric]
            )
            effectiveness_scores[platform] = score

        return {
            'platform_stats': platform_stats.to_dict(),
            'normalized_metrics': normalized_metrics,
            'effectiveness_scores': effectiveness_scores,
            'best_platform': max(effectiveness_scores.items(), key=lambda x: x[1])[0]
        }

    def analyze_content_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in content performance."""
        # Add content analysis results to data
        content_optimizer = ContentOptimizer()
        self.data['content_metrics'] = self.data['content'].apply(
            content_optimizer.analyze_content
        )

        patterns = {
            'sentiment_correlation': {},
            'length_correlation': {},
            'hashtag_correlation': {},
            'time_correlation': {}
        }

        for platform in self.data['platform'].unique():
            platform_data = self.data[self.data['platform'] == platform]
            
            # Extract metrics from content_metrics
            sentiment_values = platform_data['content_metrics'].apply(
                lambda x: x['sentiment']
            )
            
            # Calculate correlations
            for metric in self.metrics:
                patterns['sentiment_correlation'][f"{platform}_{metric}"] = stats.pearsonr(
                    sentiment_values,
                    platform_data[metric]
                )[0]

        return patterns

    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate platform-specific recommendations."""
        patterns = self.analyze_content_patterns()
        platform_stats = self.compare_platforms()
        
        recommendations = {}
        
        for platform in self.data['platform'].unique():
            platform_recs = []
            
            # Analyze performance patterns
            high_performing_posts = self.data[
                (self.data['platform'] == platform) &
                (self.data['engagement'] > self.data['engagement'].mean())
            ]
            
            # Generate recommendations based on patterns
            if len(high_performing_posts) > 0:
                common_traits = self._analyze_post_traits(high_performing_posts)
                platform_recs.extend(self._generate_trait_recommendations(common_traits))
            
            # Add timing recommendations
            timing_analysis = self._analyze_posting_times(platform)
            platform_recs.extend(timing_analysis)
            
            recommendations[platform] = platform_recs

        return recommendations

    def _analyze_post_traits(self, posts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze common traits of high-performing posts."""
        traits = {
            'avg_length': posts['content'].str.len().mean(),
            'avg_hashtags': posts['content'].str.count('#').mean(),
            'common_words': self._get_common_words(posts['content']),
            'sentiment': posts['content_metrics'].apply(lambda x: x['sentiment']).mean()
        }
        return traits

    def _generate_trait_recommendations(self, traits: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on content traits."""
        recs = []
        
        if traits['avg_length'] > 100:
            recs.append("Longer posts tend to perform better. Aim for detailed content.")
        else:
            recs.append("Shorter, concise posts show better engagement.")
            
        if traits['avg_hashtags'] > 5:
            recs.append("Use multiple relevant hashtags (5+) to increase reach.")
        
        if traits['sentiment'] > 0.2:
            recs.append("Positive sentiment correlates with higher engagement.")
            
        return recs

    def _analyze_posting_times(self, platform: str) -> List[str]:
        """Analyze optimal posting times."""
        platform_data = self.data[self.data['platform'] == platform]
        
        # Group by hour and calculate average engagement
        platform_data['hour'] = platform_data['timestamp'].dt.hour
        hourly_performance = platform_data.groupby('hour')['engagement'].mean()
        
        best_hours = hourly_performance.nlargest(3).index.tolist()
        
        return [
            f"Best posting times: {', '.join([f'{hour}:00' for hour in best_hours])}"
        ]

# Update main SocialMediaBot class
class SocialMediaBot:
    def __init__(self, config_path: str, templates_dir: str, analytics_dir: str):
        # ... (previous initialization code) ...
        self.ab_testing = ABTest()
        self.content_generator = ContentGenerator(config['openai_api_key'])
        self.cross_platform_analytics = None  # Will be initialized after data collection

    async def schedule_post_with_variations(self, content: str, platform: str, 
                                         scheduled_time: datetime,
                                         num_variations: int = 2,
                                         test_duration_days: int = 7):
        """Schedule post with A/B testing variations."""
        variations = self.content_generator.generate_variations(content, num_variations)
        
        # Create A/B test
        variants = [{'id': f'variant_{i}', 'content': v} 
                   for i, v in enumerate([content] + variations)]
        
        test_id = self.ab_testing.create_test(
            f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            variants,
            'engagement'
        )

        # Schedule variants
        time_slots = [
            scheduled_time + timedelta(minutes=i*30) 
            for i in range(len(variants))
        ]

        for variant, post_time in zip(variants, time_slots):
            self.schedule_post(
                variant['content'],
                platform,
                post_time,
                test_id=test_id,
                variant_id=variant['id']
            )

        # Schedule test analysis
        analysis_time = scheduled_time + timedelta(days=test_duration_days)
        self.schedule_task(
            analysis_time,
            self.analyze_ab_test,
            test_id
        )

    def analyze_ab_test(self, test_id: str):
        """Analyze A/B test results and update content strategies."""
        results = self.ab_testing.analyze_test(test_id)
        
        if results.get('winner'):
            winning_content = next(
                v['content'] for v in self.ab_testing.tests[test_id]['variants']
                if v['id'] == results['winner']
            )
            
            # Update content optimization strategies
            self.content_generator.update_optimization_rules(
                self.content_optimizer.analyze_content(winning_content)
            )

    def update_analytics(self):
        """Update analytics including cross-platform analysis."""
        super().update_analytics()
        
        # Initialize cross-platform analytics with collected data
        self.cross_platform_analytics = CrossPlatformAnalytics(
            [post.analytics for post in self.posts_queue if post.analytics]
        )
        
        # Generate and store recommendations
        recommendations = self.cross_platform_analytics.generate_recommendations()
        
        # Save recommendations to file
        with open(os.path.join(self.analytics_dir, 'recommendations.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
