// Copyright 2024 The SRNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"

	"github.com/pointlander/datum/bible"
)

const (
	// NumberOfVerses is the number of verses in the bible
	NumberOfVerses = 31102
	// Symbols is the number of symbols
	Symbols = 256
	// Space is the state space of the Println
	Space = 256
	// Width is the width of the neural network
	Width = Symbols + Space
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .0001
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Markov computes a markov model of the bible
func Markov(verses []bible.Verse) *[256][256][256]float32 {
	markov := [256][256][256]float32{}
	for _, verse := range verses {
		a, b := 0, 0
		for _, v := range verse.Verse {
			markov[a][b][v]++
			a, b = int(v), a
		}
	}
	for i := range markov {
		for j := range markov[i] {
			sum := float32(0.0)
			for _, v := range markov[i][j] {
				sum += v
			}
			if sum > 0 {
				for k := range markov[i][j] {
					markov[i][j][k] /= sum
				}
			}
		}
	}
	return &markov
}

func main() {
	rng := rand.New(rand.NewSource(1))
	_ = rng

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	markov := Markov(verses)
	_ = markov
}
