from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    pipeline
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from profanity_check import predict_prob
import spacy
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing_extensions import TypedDict
import asyncio
from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
from msrest.authentication import CognitiveServicesCredentials

class VisualContent(TypedDict):
    image_path: str
    analysis: Dict[str, Any]
    safety_score: float
    brand_compliance: bool
    optimization_suggestions: List[str]

class AudienceSentiment(TypedDict):
    overall_score: float
    trend: List[float]
    topics: Dict[str, float]
    key_concerns: List[str]
    recommendations: List[str]

class Report(TypedDict):
    title: str
    sections: List[Dict[str, str]]
    metrics: Dict[str, float]
    visualizations: List[str]
    recommendations: List[str]

class VisualAnalyzer:
    def __init__(self, config: Dict):
        """Initialize visual content analysis tools."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        ).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.object_detector = pipeline("object-detection")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    async def analyze_image(self, image_path: str) -> VisualContent:
        """Comprehensive analysis of visual content."""
        image = Image.open(image_path)
        cv_image = cv2.imread(image_path)

        analysis_tasks = [
            self._generate_caption(image),
            self._detect_objects(image),
            self._analyze_composition(cv_image),
            self._check_brand_compliance(image),
            self._generate_optimization_suggestions(image)
        ]

        results = await asyncio.gather(*analysis_tasks)
        caption, objects, composition, compliance, suggestions = results

        return VisualContent(
            image_path=image_path,
            analysis={
                "caption": caption,
                "objects": objects,
                "composition": composition
            },
            safety_score=self._calculate_safety_score(objects, composition),
            brand_compliance=compliance,
            optimization_suggestions=suggestions
        )

    async def _generate_caption(self, image: Image) -> str:
        """Generate descriptive caption for image."""
        pixel_values = self.feature_extractor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        output_ids = self.image_model.generate(
            pixel_values,
            max_length=50,
            num_beams=4
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _detect_objects(self, image: Image) -> List[Dict]:
        """Detect and analyze objects in image."""
        results = self.object_detector(image)
        return [
            {
                "label": r["label"],
                "score": r["score"],
                "box": r["box"]
            }
            for r in results
        ]

class ContentModerator:
    def __init__(self, config: Dict):
        """Initialize content moderation tools."""
        self.text_classifier = pipeline("text-classification")
        self.profanity_detector = predict_prob
        self.moderator_client = ContentModeratorClient(
            endpoint=config["moderator_endpoint"],
            credentials=CognitiveServicesCredentials(config["moderator_key"])
        )
        self.nlp = spacy.load("en_core_web_sm")

    async def moderate_content(self, content: Dict) -> Dict[str, Any]:
        """Moderate content across multiple dimensions."""
        text = content.get("text", "")
        image_path = content.get("image_path")

        moderation_tasks = [
            self._check_text_safety(text),
            self._analyze_tone(text),
            self._check_compliance(text)
        ]

        if image_path:
            moderation_tasks.append(self._moderate_image(image_path))

        results = await asyncio.gather(*moderation_tasks)
        
        return {
            "safety_score": results[0],
            "tone_analysis": results[1],
            "compliance_check": results[2],
            "image_moderation": results[3] if image_path else None,
            "requires_review": any(r.get("requires_review", False) for r in results),
            "suggestions": self._generate_suggestions(results)
        }

    async def _check_text_safety(self, text: str) -> Dict[str, Any]:
        """Check text for safety concerns."""
        profanity_score = self.profanity_detector([text])[0]
        classification = self.text_classifier(text)[0]
        
        return {
            "profanity_score": profanity_score,
            "classification": classification,
            "requires_review": profanity_score > 0.7
        }

class SentimentTracker:
    def __init__(self):
        """Initialize sentiment tracking tools."""
        self.analyzer = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.topic_model = BERTopic()

    async def track_sentiment(self, comments: List[str]) -> AudienceSentiment:
        """Track and analyze audience sentiment."""
        sentiments = [self.analyzer.polarity_scores(comment) for comment in comments]
        
        # Topic extraction and sentiment per topic
        topics, _ = self.topic_model.fit_transform(comments)
        topic_sentiments = self._calculate_topic_sentiments(comments, topics)

        # Trend analysis
        sentiment_trend = self._analyze_sentiment_trend(sentiments)

        # Generate recommendations
        recommendations = self._generate_recommendations(topic_sentiments, sentiment_trend)

        return AudienceSentiment(
            overall_score=np.mean([s["compound"] for s in sentiments]),
            trend=sentiment_trend,
            topics=topic_sentiments,
            key_concerns=self._identify_key_concerns(comments, sentiments),
            recommendations=recommendations
        )

    def _calculate_topic_sentiments(
        self, comments: List[str], topics: List[int]
    ) -> Dict[str, float]:
        """Calculate sentiment scores per topic."""
        topic_sentiments = defaultdict(list)
        
        for comment, topic in zip(comments, topics):
            sentiment = self.analyzer.polarity_scores(comment)["compound"]
            topic_sentiments[str(topic)].append(sentiment)
            
        return {
            topic: np.mean(scores)
            for topic, scores in topic_sentiments.items()
        }

class ReportGenerator:
    def __init__(self):
        """Initialize report generation tools."""
        self.pdf = FPDF()
        self.nlp = spacy.load("en_core_web_sm")

    async def generate_report(self, data: Dict[str, Any]) -> Report:
        """Generate comprehensive performance report."""
        # Create report sections
        sections = await asyncio.gather(
            self._create_executive_summary(data),
            self._create_performance_analysis(data),
            self._create_audience_insights(data),
            self._create_content_analysis(data),
            self._create_recommendations(data)
        )

        # Generate visualizations
        visualizations = await self._create_visualizations(data)

        return Report(
            title=f"Social Media Performance Report - {datetime.now().strftime('%Y-%m-%d')}",
            sections=sections,
            metrics=self._extract_key_metrics(data),
            visualizations=visualizations,
            recommendations=self._generate_strategic_recommendations(data)
        )

    async def _create_visualizations(self, data: Dict[str, Any]) -> List[str]:
        """Create data visualizations for the report."""
        visualization_paths = []

        # Engagement trends
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data["engagement_over_time"])
        plt.title("Engagement Trends")
        plt.savefig("engagement_trend.png")
        visualization_paths.append("engagement_trend.png")

        # Sentiment distribution
        plt.figure(figsize=(8, 8))
        sns.pieplot(data=data["sentiment_distribution"])
        plt.title("Audience Sentiment Distribution")
        plt.savefig("sentiment_dist.png")
        visualization_paths.append("sentiment_dist.png")

        # Word cloud
        wordcloud = WordCloud().generate_from_frequencies(data["topic_frequencies"])
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("wordcloud.png")
        visualization_paths.append("wordcloud.png")

        return visualization_paths

class EnhancedSocialMediaBot:
    def __init__(self, config_path: str):
        """Initialize enhanced bot with new features."""
        super().__init__(config_path)
        self.visual_analyzer = VisualAnalyzer(self.config)
        self.content_moderator = ContentModerator(self.config)
        self.sentiment_tracker = SentimentTracker()
        self.report_generator = ReportGenerator()

    async def process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process content through all analysis pipelines."""
        tasks = [
            self.visual_analyzer.analyze_image(content["image_path"])
            if "image_path" in content else None,
            self.content_moderator.moderate_content(content),
            self.sentiment_tracker.track_sentiment(content.get("comments", [])),
        ]

        results = await asyncio.gather(*[t for t in tasks if t is not None])
        
        return {
            "visual_analysis": results[0] if "image_path" in content else None,
            "moderation_results": results[1],
            "sentiment_analysis": results[2],
        }

    async def generate_performance_report(self, data: Dict[str, Any]) -> Report:
        """Generate comprehensive performance report."""
        return await self.report_generator.generate_report(data)

async def main():
    """Example usage of enhanced features."""
    config = {
        "moderator_endpoint": "your_endpoint",
        "moderator_key": "your_key"
    }
    
    bot = EnhancedSocialMediaBot("config.json")
    
    # Process content
    content = {
        "text": "Check out our new product!",
        "image_path": "product.jpg",
        "comments": [
            "Great product, loving it!",
            "Could be better...",
            "Amazing features!"
        ]
    }
    
    results = await bot.process_content(content)
    
    # Generate report
    report = await bot.generate_performance_report({
        "content_results": results,
        "engagement_over_time": pd.DataFrame(...),  # Your engagement data
        "sentiment_distribution": pd.DataFrame(...),  # Your sentiment data
        "topic_frequencies": {...}  # Your topic frequency data
    })
    
    print(f"Report generated: {report['title']}")

if __name__ == "__main__":
    asyncio.run(main())
