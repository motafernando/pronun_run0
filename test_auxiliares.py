import unittest
from poc import encode_auxiliary, ATTR_HEADS


class AuxiliaryTests(unittest.TestCase):
    def test_nested_spans_have_independent_properties(self):
        offsets=[(0,4),(5,10),(11,15)]
        spans=[dict(label='Encaminhamento',begin=0,end=15,properties=['Intimacao']),
               dict(label='Prazo',begin=5,end=10,properties=['Explicito'])]
        r=encode_auxiliary(offsets,spans)
        self.assertTrue(all(x==ATTR_HEADS['Encaminhamento'].index('Intimacao') for x in r['Encaminhamento']))
        self.assertEqual(r['Prazo'],[-100,ATTR_HEADS['Prazo'].index('Explicito'),-100])

    def test_optional_missing_and_conflicting_are_masked(self):
        spans=[dict(label='Encaminhamento',begin=0,end=4,properties=[]),
               dict(label='Prazo',begin=5,end=10,properties=['Explicito','Condicionado'])]
        r=encode_auxiliary([(0,4),(5,10)],spans)
        self.assertEqual(r['Encaminhamento'],[-100,-100])
        self.assertEqual(r['Prazo'],[-100,-100])

    def test_curated_encaminhamento_values_are_trainable(self):
        spans=[dict(label='Encaminhamento',begin=0,end=4,properties=['Despacho']),
               dict(label='Encaminhamento',begin=5,end=9,properties=['DecisaoDeInstanciaSuperior'])]
        r=encode_auxiliary([(0,4),(5,9)],spans)
        self.assertEqual(r['Encaminhamento'],[
            ATTR_HEADS['Encaminhamento'].index('Despacho'),
            ATTR_HEADS['Encaminhamento'].index('DecisaoDeInstanciaSuperior')])


if __name__=='__main__':unittest.main()
