package main

import "testing"

func TestClassMergePromotesTouchingSourceComponent(t *testing.T) {
	w, h := 8, 4
	m := make([]uint16, w*h)
	for y := 1; y <= 2; y++ {
		for x := 1; x <= 2; x++ {
			m[y*w+x] = 1 // building
		}
		for x := 3; x <= 5; x++ {
			m[y*w+x] = 2 // pavilion
		}
	}
	applyClassMergeRules(m, w, h, [6]float64{0, 0.2, 0, 0, 0, -0.2}, "PROJCRS[\"test\"]", []ClassMergeRule{{
		TargetClass:   0,
		SourceClasses: []int{1},
	}})
	for y := 1; y <= 2; y++ {
		for x := 1; x <= 5; x++ {
			if got := m[y*w+x]; got != 1 {
				t.Fatalf("pixel (%d,%d)=%d, want target encoding 1", x, y, got)
			}
		}
	}
}

func TestClassMergeKeepsDetachedSource(t *testing.T) {
	w, h := 10, 3
	m := make([]uint16, w*h)
	m[1*w+1] = 1
	m[1*w+8] = 2
	applyClassMergeRules(m, w, h, [6]float64{0, 1, 0, 0, 0, -1}, "PROJCRS[\"test\"]", []ClassMergeRule{{
		TargetClass:   0,
		SourceClasses: []int{1},
		DistanceM:     0.3,
	}})
	if got := m[1*w+8]; got != 2 {
		t.Fatalf("detached source=%d, want 2", got)
	}
}

func TestClassMergeRulesRelevant(t *testing.T) {
	rules := []ClassMergeRule{{TargetClass: 0, SourceClasses: []int{1}, DistanceM: 0.3}}
	if !classMergeRulesRelevant(rules, []int{0}) {
		t.Fatal("target class search must apply merge rules")
	}
	if !classMergeRulesRelevant(rules, []int{1}) {
		t.Fatal("source class search must apply merge rules")
	}
	if classMergeRulesRelevant(rules, []int{2}) {
		t.Fatal("unrelated class search must skip merge rules")
	}
}
