package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/icza/mjpeg"
)

func main() {
	aw, err := mjpeg.New("test.avi", 1200, 800, 60)
	if err != nil {
		panic(err)
	}

	entries, err := os.ReadDir("jpgs/")
	if err != nil {
		fmt.Printf("Could not read directory: %v\n", err)
		os.Exit(1)
	}

	sort.Slice(entries, func(i, j int) bool {
		n1 := strings.Split(entries[i].Name(), ".j")[0]
		n2 := strings.Split(entries[j].Name(), ".j")[0]
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
		//data, err := os.ReadFile(filepath.Join("jpgs/", file.Name()))
		data, err := os.ReadFile(filepath.Join("jpgs/", file.Name()))
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
