# WP-0001 - HKG Tmax data acquisition reset

## Purpose

This work package replaces the repeated Daily Extract polling experiment loop
with a durable weather-data acquisition program for HKO Tmax forecasting.

## Scope

- weather data acquisition, archiving, normalization scaffolding, provenance,
  coverage, and QC;
- no Polymarket work;
- no modelling or forecast scoring;
- no settlement-parity work;
- no rapid Daily Extract polling.

## Current Status

In progress. The first coherent milestone is the acquisition reset: data-root
selection, content-addressed raw storage, append-only manifests, collector
schedules, Windows scripts, source catalog, initial HKO acquisition batch, and
reports.

## Completion Boundary

This work package remains active until the acquisition goal's completion gate is
met: all required source families classified with evidence, accessible P0
sources implemented and backfilled or prospectively collected, P1 sources
acquired or blocked with activation plans, reports current, collectors durable,
and final checks passing.
