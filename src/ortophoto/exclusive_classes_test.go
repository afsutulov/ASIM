package main

import "testing"

func TestAssignClassPixelKeepsSingleWinner(t *testing.T) {
	classes := make([]uint16, 1)
	scores := make([]float32, 1)
	if !assignClassPixel(classes, scores, 0, 0, 0.40) {
		t.Fatal("first class was not written")
	}
	if assignClassPixel(classes, scores, 0, 2, 0.30) {
		t.Fatal("weaker class unexpectedly replaced winner")
	}
	if classes[0] != 1 {
		t.Fatalf("expected encoded class 1, got %d", classes[0])
	}
	if assignClassPixel(classes, scores, 0, 5, 0.80) {
		t.Fatal("replacement must not be counted as a new foreground pixel")
	}
	if classes[0] != 6 || scores[0] != 0.80 {
		t.Fatalf("stronger class did not win: class=%d score=%f", classes[0], scores[0])
	}
}
