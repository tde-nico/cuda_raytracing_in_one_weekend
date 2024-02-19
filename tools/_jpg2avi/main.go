package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/icza/mjpeg"
)

func main() {
	var (
		inDir string
		out   string
		w     int
		h     int
		fps   int
	)

	flag.StringVar(&inDir, "i", "jpgs", "Input directory")
	flag.StringVar(&out, "o", "out.avi", "Output file")
	flag.IntVar(&w, "w", 1200, "Width of the video")
	flag.IntVar(&h, "h", 800, "Height of the video")
	flag.IntVar(&fps, "fps", 60, "Frames per second")
	flag.Parse()

	aw, err := mjpeg.New(out, int32(w), int32(h), int32(fps))
	if err != nil {
		panic(err)
	}

	entries, err := os.ReadDir(inDir)
	if err != nil {
		fmt.Printf("Could not read directory: %v\n", err)
		os.Exit(1)
	}

	sort.Slice(entries, func(i, j int) bool {
		n1 := strings.Split(entries[i].Name(), ".jpg")[0]
		n2 := strings.Split(entries[j].Name(), ".jpg")[0]
		v1, err := strconv.ParseFloat(n1, 32)
		if err != nil {
			panic(err)
		}
		v2, err := strconv.ParseFloat(n2, 32)
		if err != nil {
			panic(err)
		}
		return v1 < v2
	})

	for _, file := range entries {
		data, err := os.ReadFile(filepath.Join(inDir, file.Name()))
		if err != nil {
			panic(err)
		}
		err = aw.AddFrame(data)
		if err != nil {
			panic(err)
		}
	}
	err = aw.Close()
	if err != nil {
		panic(err)
	}
}
