-- BFR1: Schema definition
-- Database Module: BPC Dataset Schema
-- Dataset: ibm-research/BPC (3,077 rows)

CREATE DATABASE IF NOT EXISTS bpc_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE bpc_db;

CREATE TABLE IF NOT EXISTS bpc_dataset (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    phrase      TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      VARCHAR(10) NOT NULL,
    qid         VARCHAR(20),
    situation   INT,
    category    VARCHAR(20),
    domain      VARCHAR(50),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);