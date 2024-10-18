# Predicting Creative Effectiveness and Sentiment Analysis for Ad Campaigns

## Overview
This project aims to develop an advanced machine learning system to predict the effectiveness of ad creatives (images or videos) based on their attributes and perform visual sentiment analysis to evaluate their emotional impact. The system will provide actionable insights for optimizing ad campaigns. The proposed objectives are meant to be guidelines for the project, and the final implementation may include additional features and functionalities.


## Dataset: 
The dataset used in this project encompasses detailed information on various ad campaigns, specifically focusing on different attributes of the ads, their performance metrics, and associated metadata. Each row in the dataset represents an individual ad campaign instance with the following features:

- **campaign_item_id**: Unique identifier for each campaign item.
- **no_of_days**: Duration in days the ad campaign ran.
- **time**: The date on which the ad campaign was executed.
- **ext_service_id**: Identifier for the external service used for the ad campaign.
- **ext_service_name**: Name of the external service platform (e.g., Facebook Ads, Google Ads).
- **creative_id**: Identifier for the creative content used in the ad.
- **search_tags**: Tags associated with the ad for search optimization.
- **template_id**: Identifier for the ad template used.
- **landing_page**: URL of the landing page linked to the ad.
- **advertiser_id**: Unique identifier for the advertiser.
- **advertiser_name**: Name of the advertiser.
- **network_id**: Identifier for the ad network.
- **approved_budget**: Budget approved for the ad campaign in the advertiser's currency.
- **advertiser_currency**: Currency used by the advertiser.
- **channel_id**: Identifier for the ad channel used.
- **channel_name**: Name of the ad channel (e.g., Mobile, Social, Video).
- **max_bid_cpm**: Maximum bid for cost per thousand impressions.
- **network_margin**: Margin of the ad network.
- **campaign_budget_usd**: Budget for the campaign in USD.
- **impressions**: Number of times the ad was displayed.
- **clicks**: Number of clicks the ad received.
- **stats_currency**: Currency used for the ad performance stats.
- **currency_code**: Code for the currency.
- **exchange_rate**: Exchange rate used for currency conversion.
- **media_cost_usd**: Media cost of the ad in USD.
- **search_tag_cat**: Category of the search tag.
- **cmi_currency_code**: Currency code used in cost per impression.
- **timezone**: Timezone of the ad campaign.
- **weekday_cat**: Category indicating if the campaign was run on a weekday or weekend.
- **keywords**: Keywords associated with the ad campaign.

This dataset provides a comprehensive view of ad campaign parameters and their corresponding performance metrics, which will be instrumental in developing models for predicting creative effectiveness and conducting sentiment analysis. The inclusion of metadata such as search tags, landing pages, advertiser details, and keywords further enriches the dataset, enabling more nuanced analysis and feature extraction.


## Objectives
1. **Feature Extraction and Analysis**:
   - Extract and analyze visual features from images and videos using computer vision techniques.
      - For images, examples would be:
         - color 
         - texture
         - shape
         - composition
         - object detection (i.e. number of objects, predominant objects, etc.)
      - For videos, examples would be:
         - keyframes
         - motion
         - scene transitions
         - object tracking
         - audio analysis
   - Integrate metadata (e.g., captions, tags, performance metrics) into the model.

2. **Multi-Modal Model Development**:
   - Combine visual and textual data for more accurate predictions using state-of-the-art models.

3. **Visual Sentiment Analysis**:
   - Analyze visual content and detect emotions using deep learning models.
   - Integrate sentiment analysis for text to provide a comprehensive sentiment score.
   - Include sentiment analysis at a creative level (i.e., inside the image/video).

4. **Performance Prediction**:
   - Train models to predict key performance metrics, such as the number of clicks, based on extracted features.
   - Develop an evaluation framework to measure model accuracy.
   - Highlight feature importance to understand which extracted features contribute most to the predictions.

5. **Pipeline Development**:
   - Build a robust pipeline for data ingestion, feature engineering, model training, and validation.

6. **Actionable Insights**:
   - Provide actionable insights and confidence scores for downstream usage.


## Requirements
- Python 3.7+
- TensorFlow or PyTorch
- Any other libraries you may need

## Deliverables
- **Image Analysis Pipeline**:
  - A comprehensive Jupyter notebook or Python script that includes all the steps for data preprocessing, feature extraction, and initial data analysis.
  - The script will demonstrate how to extract visual features from images and videos, such as color, texture, shape, composition, keyframes, motion, scene transitions, and object detection.
  - It will also show how to integrate metadata (e.g., captions, tags, performance metrics) into the feature set.
  - Detailed comments and explanations will be provided within the code to ensure clarity and ease of understanding.

- **Forecasting Model Development (Model Selection & Evaluation) Code with Detailed Documentation**:
  - A well-documented codebase for the development of multi-modal models that combine visual and textual data for predicting the effectiveness of ad creatives.
  - The code will include the implementation of visual sentiment analysis models to detect emotions from visual content and text.
  - Scripts for training models to predict key performance metrics, such as the number of clicks, based on the extracted features.
  - An evaluation framework to measure the accuracy and performance of the developed models.
  - Feature importance analysis to understand the impact of each extracted feature on the predictions.
  - Detailed documentation will accompany the code, explaining the model selection process, hyperparameter tuning, and evaluation metrics used.
  - Examples and guidelines for replicating the experiments and extending the models with additional features and functionalities.

- **Summary of Insights and Recommendations**:
  - A summary report that highlights the actionable insights derived from the analysis and model predictions.
  - Recommendations for optimizing ad campaigns based on the model predictions and sentiment analysis results.
  - Visualizations and explanations of the key findings, including feature importance, sentiment analysis results, and performance predictions.
  - A detailed explanation of the methodology used, the challenges faced, and the potential improvements for future iterations.

These deliverables will provide a solid foundation for understanding the workflow, from data extraction and feature engineering to model training and evaluation, enabling users to effectively predict ad creative performance, analyze sentiment, and identify key features driving the performance.