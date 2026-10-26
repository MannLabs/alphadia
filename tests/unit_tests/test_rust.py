"""Smoke tests for the compiled Rust extension `alphadia._rust`."""

from alphadia._rust import hello_world


def test_hello_world_returns_greeting_from_rust():
    # given

    # when
    greeting = hello_world()

    # then
    assert greeting == "Hello from alphadia Rust!"
