// Copyright 2024 The SRNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/compress"
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
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

var (
	// FlagLearn learn mode
	FlagLearn = flag.String("learn", "", "learn mode")
	// FlagInference inference mode
	FlagInference = flag.String("infer", "", "inference mode")
	// FlagFile is a weights file
	FlagFile = flag.String("file", "", "weights file")
)

// Context is a markov model context
type Context [2]rune

// SymbolMap maps symbols
type SymbolMap struct {
	Map   map[rune]int
	Inv   map[int]rune
	Width int
}

// NewSymbolMap creates a new symbol map
func NewSymbolMap(verses []bible.Verse) SymbolMap {
	m, i, width := make(map[rune]int), make(map[int]rune), 0
	m[0] = width
	i[width] = 0
	width++
	files := []string{
		"10.txt.utf-8",
		"1342.txt.utf-8",
		"145.txt.utf-8",
		"1513.txt.utf-8",
		"2641.txt.utf-8",
		"2701.txt.utf-8",
		"37106.txt.utf-8",
		"84.txt.utf-8",
	}
	for _, book := range files {
		book = "books/" + book
		fmt.Println(book)
		data, err := os.ReadFile(book)
		if err != nil {
			panic(err)
		}
		verse := string(data)

		for _, s := range verse {
			if _, ok := m[s]; !ok {
				m[s] = width
				i[width] = s
				width++
			}
		}
	}
	return SymbolMap{
		Map:   m,
		Inv:   i,
		Width: width,
	}
}

// Markov computes a markov model of the bible
func (s SymbolMap) Markov(verses []bible.Verse) [][][]float32 {
	markov := make([][][]float32, s.Width)
	for i := range markov {
		markov[i] = make([][]float32, s.Width)
		for j := range markov[i] {
			markov[i][j] = make([]float32, s.Width)
		}
	}
	for _, verse := range verses {
		a, b := rune(0), rune(0)
		for _, v := range verse.Verse {
			markov[s.Map[a]][s.Map[b]][s.Map[v]]++
			a, b = v, a
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
	return markov
}

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

func main() {
	flag.Parse()

	if *FlagInference == "srnn" {
		Inference()
		return
	} else if *FlagInference == "compress" {
		CompressInference()
		return
	} else if *FlagLearn == "srnn" {
		Learn()
		return
	} else if *FlagLearn == "markov" {
		Markov()
		return
	} else if *FlagLearn == "compress" {
		Compress()
		return
	}
}

// CompressInference inference compression mode
func CompressInference() {
	seed := int64(1)
	rng := rand.New(rand.NewSource(seed))

	input := tf64.NewV(256, 1)
	input.X = input.X[:cap(input.X)]

	name := *FlagFile
	set := tf64.NewSet()
	cost, epoch, err := set.Open(name)
	if err != nil {
		panic(err)
	}
	fmt.Println(name, cost, epoch)

	t := 1.0
	temp := tf64.NewV(256, 1)
	for i := 0; i < 256; i++ {
		temp.X = append(temp.X, 1/t)
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l2 := tf64.Softmax(tf64.Hadamard(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")), temp.Meta()))

	factory := compress.NewCDF16(2, false)
	cdf := factory(256)
	initial := "And the LORD said"
	for _, v := range initial {
		cdf.Update(uint16(v))
	}
	for i := 0; i < 4*128; i++ {
		model := cdf.Model()
		for k := range input.X {
			input.X[k] = float64(model[k+1]-model[k]) / float64(compress.CDF16Scale)
		}
		symbols := []float64{}
		selected, sum := rng.Float64(), 0.0
		l2(func(a *tf64.V) bool {
			symbols = a.X
			return true
		})
		v := rune(0)
		for j := 0; j < 256; j++ {
			sum += symbols[j]
			if sum > selected {
				v = rune(j)
				print(string(v))
				break
			}
		}
		cdf.Update(uint16(v))
	}
	fmt.Println()
}

// Compress learns a compression based model
func Compress() {
	seed := int64(1)
	rng := rand.New(rand.NewSource(seed))

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()

	input := tf64.NewV(256, 1)
	input.X = input.X[:cap(input.X)]
	output := tf64.NewV(256, 1)
	output.X = output.X[:cap(output.X)]

	set := tf64.NewSet()
	set.Add("w1", 256, 256)
	set.Add("b1", 256)
	set.Add("w2", 256, 256)
	set.Add("b2", 256)

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
	for epoch := 0; epoch < 7; epoch++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(epoch+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := 0.0
		for x, verse := range verses {
			factory := compress.NewCDF16(2, false)
			cdf := factory(256)
			set.Zero()
			cost := 0.0
			for _, v := range verse.Verse {
				model := cdf.Model()
				for k := range input.X {
					input.X[k] = float64(model[k+1]-model[k]) / float64(compress.CDF16Scale)
				}
				for k := range output.X {
					output.X[k] = 0
				}
				output.X[v] = 1
				cost += tf64.Gradient(loss).X[0]
				cdf.Update(uint16(v))
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

func Markov() {
	const ContextLength = 256

	// markov multivariate
	rng := rand.New(rand.NewSource(1))
	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	sm := NewSymbolMap(verses)
	markov := make(map[Context][][]float32)
	files := []string{
		"10.txt.utf-8",
		"1342.txt.utf-8",
		"145.txt.utf-8",
		"1513.txt.utf-8",
		"2641.txt.utf-8",
		"2701.txt.utf-8",
		"37106.txt.utf-8",
		"84.txt.utf-8",
	}
	for _, book := range files {
		book = "books/" + book
		fmt.Println(book)
		data, err := os.ReadFile(book)
		if err != nil {
			panic(err)
		}
		verse := string(data)
		ctxt := Context{}
		index, context, buffer := 0, make([]float32, sm.Width), make([]rune, ContextLength)
		for i := range buffer {
			buffer[i] = -1
		}
		total := 0
		learn := func(v rune, ctxt Context) {
			l1 := markov[ctxt]
			if l1 == nil {
				l1 = make([][]float32, sm.Width)
			}
			l2 := l1[sm.Map[v]]
			if l2 == nil {
				l2 = make([]float32, sm.Width)
			}
			for i := range l2 {
				if total > 0 {
					l2[i] += context[i] / float32(total)
				} else {
					l2[i] += context[i]
				}
			}
			l1[sm.Map[v]] = l2
			markov[ctxt] = l1
		}
		for _, v := range verse {
			learn(v, ctxt)
			cp := ctxt
			cp[1] = 0
			learn(v, cp)
			ctxt[0], ctxt[1] = v, ctxt[0]
			if total < 256 {
				total++
			} else {
				context[sm.Map[buffer[index]]]--
			}
			buffer[index] = v
			context[sm.Map[v]]++
			if _, ok := sm.Map[v]; !ok {
				panic("symbol not found")
			}
			index = (index + 1) % ContextLength
		}
	}

	ctxt := Context{}
	index, context, buffer := 0, make([]float32, sm.Width), make([]rune, ContextLength)
	for i := range buffer {
		buffer[i] = -1
	}
	total := 0
	temp := float32(1 / .01)
	initial := "The LORD said"
	for _, v := range initial {
		ctxt[0], ctxt[1] = v, ctxt[0]
		if total < 256 {
			total++
		} else {
			context[sm.Map[buffer[index]]]--
		}
		buffer[index] = v
		context[sm.Map[v]]++
		index = (index + 1) % ContextLength
	}
	for i := 0; i < 256; i++ {
		m := markov[ctxt]
		count := 0
		for j := range m {
			if m[j] != nil {
				count++
			}
		}
		if count < 1 {
			cp := ctxt
			cp[1] = 0
			m = markov[cp]
		}
		output := make([]float32, sm.Width)
		for j := range m {
			ab, aa, bb := float32(0), float32(0), float32(0)
			for k, b := range m[j] {
				a := context[k]
				ab += a * b
				aa += a * a
				bb += b * b
			}
			if aa > 0 && bb > 0 {
				output[j] = temp * ab / (float32(math.Sqrt(float64(aa)) * math.Sqrt(float64(bb))))
			}
		}
		v := rune(0)
		softmax(output)
		t := rng.Float32()
		sum := float32(0)
		for j, value := range output {
			sum += value
			if sum > t {
				v = sm.Inv[j]
				break
			}
		}
		print(string(v))
		ctxt[0], ctxt[1] = v, ctxt[0]
		if total < 256 {
			total++
		} else {
			context[sm.Map[buffer[index]]]--
		}
		buffer[index] = v
		context[sm.Map[v]]++
		index = (index + 1) % ContextLength
	}
	print("\n")
}

// Learn learns the model
func Learn() {
	seed := int64(1)
	rng := rand.New(rand.NewSource(seed))

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	sm := NewSymbolMap(verses)
	markov := sm.Markov(verses)
	sum, sumSquared := make([]float32, sm.Width), make([]float32, sm.Width*sm.Width)
	mean, stddev := make([]float32, sm.Width), make([]float32, sm.Width*sm.Width)

	input := tf64.NewV(sm.Width*sm.Width+2*sm.Width, 1)
	input.X = input.X[:cap(input.X)]
	output := tf64.NewV(sm.Width, 1)
	output.X = output.X[:cap(output.X)]

	set := tf64.NewSet()
	set.Add("w1", sm.Width*sm.Width+2*sm.Width, 4*sm.Width)
	set.Add("b1", 4*sm.Width)
	set.Add("w2", 4*sm.Width, sm.Width)
	set.Add("b2", sm.Width)

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
	for epoch := 0; epoch < 7; epoch++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(epoch+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		for i := range verses {
			j := i + rng.Intn(len(verses)-i)
			verses[i], verses[j] = verses[j], verses[i]
		}

		total := 0.0
		for x, verse := range verses {
			a, b := rune(0), rune(0)
			for y := range sum {
				sum[y] = 0
			}
			for y := range sumSquared {
				sumSquared[y] = 0
			}
			buffer, cost := []float32{}, 0.0
			set.Zero()
			for i, v := range verse.Verse {
				if buffer != nil {
					for k1, value1 := range buffer {
						sum[k1] = .9*sum[k1] + .1*value1
						for k2, value2 := range buffer {
							sumSquared[k1*sm.Width+k2] = .9*sumSquared[k1*sm.Width+k2] + .1*value1*value2
						}
					}
					n := float32(i)
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
						stddev[k] = float32(math.Sqrt(float64((value / n) - mean[k/sm.Width]*mean[k%sm.Width])))
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
					for _, value := range buffer {
						s += value
					}
					for _, value := range buffer {
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
					output.X[sm.Map[v]] = 1
					cost += tf64.Gradient(loss).X[0]
				}
				buffer = markov[sm.Map[a]][sm.Map[b]]
				a, b = v, a
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

func Inference() {
	seed := int64(1)
	rng := rand.New(rand.NewSource(seed))

	bible, err := bible.Load()
	if err != nil {
		panic(err)
	}
	verses := bible.GetVerses()
	sm := NewSymbolMap(verses)
	markov := sm.Markov(verses)
	sum, sumSquared := make([]float32, sm.Width), make([]float32, sm.Width*sm.Width)
	mean, stddev := make([]float32, sm.Width), make([]float32, sm.Width*sm.Width)

	input := tf64.NewV(sm.Width*sm.Width+2*sm.Width, 1)
	input.X = input.X[:cap(input.X)]

	name := *FlagFile
	set := tf64.NewSet()
	cost, epoch, err := set.Open(name)
	if err != nil {
		panic(err)
	}
	fmt.Println(name, cost, epoch)

	t := .5
	temp := tf64.NewV(sm.Width, 1)
	for i := 0; i < sm.Width; i++ {
		temp.X = append(temp.X, 1/t)
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l2 := tf64.Softmax(tf64.Hadamard(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")), temp.Meta()))

	a, b := 'h', 'T'
	buffer := markov[sm.Map[a]][sm.Map[b]]
	print(string(b))
	print(string(a))
	for i := 0; i < 4*128; i++ {
		for k1, value1 := range buffer {
			sum[k1] = .9*sum[k1] + .1*value1
			for k2, value2 := range buffer {
				sumSquared[k1*sm.Width+k2] = .9*sumSquared[k1*sm.Width+k2] + .1*value1*value2
			}
		}
		n := float32(i)
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
			stddev[k] = float32(math.Sqrt(float64((value / n) - mean[k/sm.Width]*mean[k%sm.Width])))
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
		for _, value := range buffer {
			s += value
		}
		for _, value := range buffer {
			if s > 0 {
				input.X[t] = float64(value / s)
			} else {
				input.X[t] = 0
			}
			t++
		}

		symbols := []float64{}
		selected, sum := rng.Float64(), 0.0
		l2(func(a *tf64.V) bool {
			symbols = a.X
			return true
		})
		v := rune(0)
		for j := 0; j < sm.Width; j++ {
			sum += symbols[j]
			if sum > selected {
				v = sm.Inv[j]
				print(string(v))
				break
			}
		}

		buffer = markov[sm.Map[a]][sm.Map[b]]
		a, b = v, a
	}
	print("\n")
}
