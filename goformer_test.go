package goformer

import "testing"

func TestHello(t *testing.T) {
	got := Hello()
	want := "Hello, goformer!"
	if got != want {
		t.Errorf("Hello() = %q, want %q", got, want)
	}
}
