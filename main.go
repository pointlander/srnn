// Copyright 2024 The SRNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/datum/bible"
	"github.com/pointlander/gradient/tf64"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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
	Eta = .01
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
	seed := int64(1)
	rng := rand.New(rand.NewSource(seed))

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	sym, next := make(map[rune]int), 0
	for _, verse := range verses {
		for _, s := range verse.Verse {
			if _, ok := sym[s]; !ok {
				sym[s] = next
				next++
			}
		}
	}
	fmt.Println(next, next*next)
	markov := Markov(verses)
	sum, sumSquared := make([]float32, Symbols), make([]float32, Symbols*Symbols)
	mean, stddev := make([]float32, Symbols), make([]float32, Symbols*Symbols)

	input := tf64.NewV(Symbols*Symbols+2*Symbols, 1)
	input.X = input.X[:cap(input.X)]
	output := tf64.NewV(Symbols, 1)
	output.X = output.X[:cap(output.X)]

	set := tf64.NewSet()
	set.Add("w1", Symbols*Symbols+2*Symbols, 4*Symbols)
	set.Add("b1", 4*Symbols)
	set.Add("w2", 4*Symbols, Symbols)
	set.Add("b2", Symbols)

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	loss := tf64.CrossEntropy(tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))), output.Meta())

	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:size]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < size; i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	points := make(plotter.XYs, 0, 8)
	epoch := 0
	total := 0.0
	pow := func(x float64) float64 {
		y := math.Pow(x, float64(epoch+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}
	for x, verse := range verses {
		a, b := 0, 0
		for y := range sum {
			sum[y] = 0
		}
		for y := range sumSquared {
			sumSquared[y] = 0
		}
		buffer, index, n := make([]*[256]float32, 256), 0, 0
		cost := 0.0
		set.Zero()
		for vv, v := range verse.Verse[:len(verse.Verse)-1] {
			buffer[index] = &markov[a][b]
			a, b = int(v), a
			if n > 0 {
				i := (index + 256 - 1) % 256
				for k1, value1 := range *buffer[i] {
					sum[k1] += value1
					for k2, value2 := range *buffer[i] {
						sumSquared[k1*256+k2] += value1 * value2
					}
				}
				n := float32(n)
				s := float32(0)
				for k, value := range sum {
					mean[k] = value / n
					s += mean[k]
				}
				t := 0
				for _, value := range mean {
					if s > 0 {
						input.X[t] = float64(value / s)
					} else {
						input.X[t] = 0
					}
					t++
				}
				s = 0
				for k, value := range sumSquared {
					stddev[k] = float32(math.Sqrt(float64((value / n) - mean[k/256]*mean[k%256])))
					s += stddev[k]
				}
				for _, value := range stddev {
					if s > 0 {
						input.X[t] = float64(value / s)
					} else {
						input.X[t] = 0
					}
					t++
				}
				s = 0
				for _, value := range buffer[index] {
					s += value
				}
				for _, value := range buffer[index] {
					if s > 0 {
						input.X[t] = float64(value / s)
					} else {
						input.X[t] = 0
					}
					t++
				}
				for k := range output.X {
					output.X[k] = 0
				}
				output.X[verse.Verse[vv+1]] = 1
				cost += tf64.Gradient(loss).X[0]
			}

			index = (index + 1) % 256
			n++
		}
		cost /= float64(len(verse.Verse))
		total += cost
		fmt.Println(x, cost)
		points = append(points, plotter.XY{X: float64(x), Y: float64(cost)})

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		if norm > 1 {
			scaling := 1 / norm
			for _, w := range set.Weights {
				if w.N == "w2" {
					continue
				}
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
		} else {
			for _, w := range set.Weights {
				if w.N == "w2" {
					continue
				}
				for l, d := range w.D {
					g := d
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
		}
	}

	err = set.Save(fmt.Sprintf("weights_%d_%d.w", seed, epoch), total, epoch)
	if err != nil {
		panic(err)
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}
}
