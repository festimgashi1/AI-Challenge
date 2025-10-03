<?php

namespace TextClassification;

class DataAnalyze
{
    public function getCategoryStats($labels)
    {
        $stats = array_count_values($labels);
        arsort($stats);
        return $stats;
    }
    
    public function getMostFrequentWords($texts, $topN = 20)
    {
        $allWords = [];
        
        foreach ($texts as $text) {
            $words = explode(' ', $text);
            foreach ($words as $word) {
                if (strlen($word) > 2) {
                    $allWords[$word] = ($allWords[$word] ?? 0) + 1;
                }
            }
        }
        
        arsort($allWords);
        return array_slice($allWords, 0, $topN, true);
    }
    
    public function printStats($dataset)
    {
        echo "=== Dataset Statistics ===\n";
        echo "Total samples: " . count($dataset['data']) . "\n";
        echo "Number of categories: " . count($dataset['categories']) . "\n";
        
        $stats = $this->getCategoryStats($dataset['labels']);
        echo "\nSamples per category:\n";
        foreach ($stats as $category => $count) {
            echo " - $category: $count samples\n";
        }
        
        $preprocessor = new TextPreprocessor();
        $processedTexts = $preprocessor->preprocessBatch(array_slice($dataset['data'], 0, 100));
        $frequentWords = $this->getMostFrequentWords($processedTexts, 15);
        
        echo "\nTop 15 most frequent words:\n";
        foreach ($frequentWords as $word => $count) {
            echo " - $word: $count occurrences\n";
        }
    }
}