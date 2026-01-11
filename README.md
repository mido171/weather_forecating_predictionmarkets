# Weather Forecasting Prediction Markets

Multi-module Java project for Epic #1 ingestion (Kalshi weather MOS + CLI).

## Modules
- models: JPA entities, enums, shared DTOs, Flyway migrations
- common: shared utilities (time, HTTP, retries, hashing)
- ingestion-service: Spring Boot app for ingestion jobs

## Build
Requires Java 18 (configured in the parent POM).

mvn clean install

## Run ingestion-service (local / in-memory)
Use the local profile to run against the in-memory H2 database.

mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--spring.profiles.active=local"

## Run ingestion-service (MySQL)
Default profile is MySQL. Update connection settings if needed.

SPRING_PROFILES_ACTIVE=mysql
SPRING_DATASOURCE_URL=jdbc:mysql://localhost:3306/weather_predictionmarkets?createDatabaseIfNotExist=true&useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC
SPRING_DATASOURCE_USERNAME=root
SPRING_DATASOURCE_PASSWORD=root

mvn -pl ingestion-service spring-boot:run
