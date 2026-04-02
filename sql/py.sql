-- ZOMATO MARKET INTELLIGENCE SQL ANALYSIS
CREATE TABLE restaurants (
  name TEXT,
  url TEXT,
  cuisines TEXT,
  area TEXT,
  timing TEXT,
  full_address TEXT,
  phonenumber TEXT,
  ishomedelivery INT,
  istakeaway INT,
  isindoorseating INT,
  isvegonly INT,
  dinner_ratings FLOAT,
  dinner_reviews INT,
  delivery_ratings FLOAT,
  delivery_reviews INT,
  knownfor TEXT,
  populardishes TEXT,
  peopleknownfor TEXT,
  averagecost INT,
  city TEXT,
  latitude FLOAT,
  longitude FLOAT,
  overall_rating FLOAT,
  total_reviews INT,
  demand_score FLOAT,
  cost_index FLOAT,
  cuisine_count INT,
  area_demand_score FLOAT,
  restaurant_density INT,
  opportunity_score FLOAT,
  area_avg_rating FLOAT
);



-- 1. Demand vs Supply
SELECT 
    area,
    COUNT(DISTINCT name) AS supply,
    SUM(total_reviews) AS demand,
    ROUND(SUM(total_reviews)*1.0 / COUNT(DISTINCT name),2) AS demand_per_restaurant
FROM restaurants
GROUP BY area
ORDER BY demand_per_restaurant DESC;


-- 2. Opportunity Score
SELECT 
    area,
    AVG(opportunity_score) AS opportunity_score
FROM restaurants
GROUP BY area
ORDER BY opportunity_score DESC
LIMIT 10;


-- 3. Top Restaurant per Area 
SELECT *
FROM (
    SELECT 
        name,
        area,
        total_reviews,
        RANK() OVER (PARTITION BY area ORDER BY total_reviews DESC) AS r
    FROM restaurants
) t
WHERE r = 1;


-- 4.High Rating, Low Reviews
SELECT 
    name,
    area,
    delivery_ratings,
    total_reviews
FROM restaurants
WHERE delivery_ratings > 4.2
AND total_reviews< 500
ORDER BY delivery_ratings DESC;


-- 5. Cuisine Demand
SELECT 
    cuisines,
    COUNT(*) AS restaurants,
    SUM(dinner_reviews + delivery_reviews) AS demand
FROM restaurants
GROUP BY cuisines
ORDER BY demand DESC
LIMIT 10;


-- 6. Market Classification
SELECT 
    area,
    COUNT(*) AS supply,
    SUM(total_reviews) AS demand,
    CASE 
       WHEN SUM(total_reviews) < 6373.4 THEN 'Saturated'
       WHEN SUM(total_reviews) > 54700.4 THEN 'High Opportunity'
        ELSE 'Moderate'
    END AS market_type
FROM restaurants
GROUP BY area;