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
The default profile uses an in-memory H2 database for local startup.

mvn -pl ingestion-service spring-boot:run

## Run ingestion-service (MySQL)
Set the MySQL profile and connection settings.

SPRING_PROFILES_ACTIVE=mysql
SPRING_DATASOURCE_URL=jdbc:mysql://localhost:3306/weather
SPRING_DATASOURCE_USERNAME=weather
SPRING_DATASOURCE_PASSWORD=weather

mvn -pl ingestion-service spring-boot:run
