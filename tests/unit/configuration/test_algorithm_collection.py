"""Unit tests for AlgorithmCollection."""
import pytest

from brisk.configuration.algorithm_collection import AlgorithmCollection

from tests.utils.factories import AlgorithmFactory

class TestAlgorithmCollectionUnit():
    def test_instantiation_no_algorithm(self):
        collection = AlgorithmCollection()
        assert collection == []

    def test_instantiation_one_algorithm(self):
        wrapper = AlgorithmFactory.simple() 
        collection = AlgorithmCollection(wrapper)
        assert collection == [wrapper]

    def test_instantiation_two_algorithm(self):
        wrapper = AlgorithmFactory.simple()
        wrapper2 = AlgorithmFactory.full(name="test_wrapper")
        collection = AlgorithmCollection(wrapper, wrapper2)
        assert collection == [wrapper, wrapper2]

    def test_append_one_algorithm(self):
        collection = AlgorithmCollection()
        wrapper = AlgorithmFactory.simple()
        collection.append(wrapper)
        assert collection == [wrapper]

    def test_append_none_algorithm_wrapper(self):
        collection = AlgorithmCollection()
        wrapper = AlgorithmFactory
        with pytest.raises(TypeError):
            collection.append(wrapper)

    def test_append_name_exists_error(self):
        collection = AlgorithmCollection()
        wrapper = AlgorithmFactory.simple()
        wrapper2 = AlgorithmFactory.simple()
        collection.append(wrapper)
        with pytest.raises(ValueError):
            collection.append(wrapper2)

    def test_list_index(self):
        wrapper = AlgorithmFactory.simple() 
        collection = AlgorithmCollection(wrapper)
        output = collection[0]
        assert output == wrapper

    def test_dict_index(self):
        wrapper = AlgorithmFactory.simple() 
        collection = AlgorithmCollection(wrapper)
        output = collection["ridge"]
        assert output == wrapper

    def test_invalid_list_index(self):
        wrapper = AlgorithmFactory.simple() 
        collection = AlgorithmCollection(wrapper)
        with pytest.raises(IndexError):
            output = collection[1]
