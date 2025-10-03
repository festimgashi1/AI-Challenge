<?php

namespace TextClassification;

class TextPreprocessor
{
    private $stopWords;
    
    public function __construct()
    {
        $this->stopWords = $this->loadStopWords();
    }
    
    private function loadStopWords()
    {
        return [
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now', 'I', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        ];
    }
    
    public function clearText($text)
    {
        $text = mb_strtolower($text, 'UTF-8');
        
        $text = preg_replace('/[^a-zA-Z\s]/', ' ', $text);
        
        $text = preg_replace('/\s+/', ' ', $text);
        $text = trim($text);
        
        return $text;
    }
    
    public function removeStopWords($text)
    {
        $words = explode(' ', $text);
        $filteredWords = array_filter($words, function($word) {
            return !in_array($word, $this->stopWords) && strlen($word) > 2;
        });
        
        return implode(' ', $filteredWords);
    }
    
    public function preprocess($text)
    {
        $text = $this->clearText($text);
        $text = $this->removeStopWords($text);
        return $text;
    }
    
    public function preprocessBatch($texts)
    {
        $processed = [];
        foreach ($texts as $text) {
            $processed[] = $this->preprocess($text);
        }
        return $processed;
    }
}