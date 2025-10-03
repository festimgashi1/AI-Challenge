<?php

namespace TextClassification;

class TfIdfVectorizer
{
    private $vocabulary = [];
    private $idf = [];
    private $fitted = false;
    
    public function fit($texts)
    {
        // Build vocabulary
        $wordCounts = [];
        $docCount = count($texts);
        
        foreach ($texts as $text) {
            $words = array_unique(explode(' ', $text));
            foreach ($words as $word) {
                if (strlen($word) > 2) {
                    $wordCounts[$word] = ($wordCounts[$word] ?? 0) + 1;
                }
            }
        }
        
        // Keep most frequent words (limit vocabulary size)
        arsort($wordCounts);
        $this->vocabulary = array_slice(array_keys($wordCounts), 0, 1000);
        $this->vocabulary = array_flip($this->vocabulary);
        
        // Calculate IDF
        foreach ($this->vocabulary as $word => $index) {
            $docFrequency = 0;
            foreach ($texts as $text) {
                $words = explode(' ', $text);
                if (in_array($word, $words)) {
                    $docFrequency++;
                }
            }
            $this->idf[$word] = $docFrequency > 0 ? log($docCount / $docFrequency) : 0;
        }
        
        $this->fitted = true;
    }
    
    public function transform($texts)
    {
        if (!$this->fitted) {
            throw new \Exception("Vectorizer must be fitted first");
        }
        
        $vectors = [];
        
        foreach ($texts as $text) {
            $words = explode(' ', $text);
            $tf = array_count_values($words);
            $totalWords = count($words);
            
            $vector = array_fill(0, count($this->vocabulary), 0);
            
            foreach ($words as $word) {
                if (isset($this->vocabulary[$word])) {
                    $index = $this->vocabulary[$word];
                    $tfValue = $tf[$word] / $totalWords;
                    $vector[$index] = $tfValue * $this->idf[$word];
                }
            }
            
            $vectors[] = $vector;
        }
        
        return $vectors;
    }
    
    public function fitTransform($texts)
    {
        $this->fit($texts);
        return $this->transform($texts);
    }
    
    public function getVocabulary()
    {
        return array_flip($this->vocabulary);
    }
    
    public function getIdf()
    {
        return $this->idf;
    }
    
    public function setVocabulary($vocabulary)
    {
        $this->vocabulary = array_flip($vocabulary);
    }
    
    public function setIdf($idf)
    {
        $this->idf = $idf;
    }
}