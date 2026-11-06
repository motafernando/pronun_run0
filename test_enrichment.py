import unittest
from enriquecimento import enrich_version, match_spans


def span(label,a,b,props=()):
    return dict(label=label,begin=a,end=b,properties=list(props),text='x'*(b-a))


class EnrichmentTest(unittest.TestCase):
    def test_alignment_preserves_boundaries_and_replaces_properties_relations(self):
        gold={'text':'x'*50,'spans':[span('Encaminhamento',0,12,['Intimacao']),span('Prazo',20,30,['Explicito'])],
              'relations':[dict(label='estabelecido',source=1,target=0)]}
        ann={'text':'x'*50,'spans':[span('Encaminhamento',1,12,['Notificacao']),span('Prazo',20,31,['Implicito'])],
             'relations':[dict(label='vinculado',source=1,target=0)]}
        out,report=enrich_version(gold,ann)
        self.assertEqual([(s['begin'],s['end']) for s in out['spans']],[(1,12),(20,31)])
        self.assertEqual(out['spans'][0]['properties'],['Intimacao'])
        self.assertEqual(out['relations'],[dict(label='estabelecido',source=1,target=0)])
        self.assertEqual(ann['spans'][0]['properties'],['Notificacao'])

    def test_ambiguous_and_different_family_are_not_forced(self):
        g=[span('Prazo',0,10),span('Prazo',0,10)]
        self.assertEqual(match_spans(g,[span('Prazo',0,10)]),{})
        self.assertEqual(match_spans([span('Prazo',0,10)],[span('Encaminhamento',0,10)]),{})

    def test_unmatched_gold_is_copied_and_extra_annotation_dropped(self):
        gold={'text':'x'*50,'spans':[span('Pronunciamento',0,10,['Despacho']),span('Prazo',20,30,['Explicito'])],
              'relations':[dict(label='estabelecido',source=1,target=0)]}
        ann={'text':'x'*50,'spans':[span('Pronunciamento',1,11,['Sentenca']),span('Encaminhamento',35,45,['Citacao'])],
             'relations':[dict(label='vinculado',source=1,target=0)]}
        out,r=enrich_version(gold,ann)
        self.assertEqual([(s['label'],s['begin'],s['end'],s['properties']) for s in out['spans']],
                         [('Pronunciamento',1,11,['Despacho']),('Prazo',20,30,['Explicito'])])
        self.assertEqual(out['relations'],gold['relations'])
        self.assertEqual(r['unmapped_annotation_spans'],1)
        self.assertEqual(r['unmapped_gold_spans'],1)

    def test_ambiguous_mapping_falls_back_to_gold_boundaries(self):
        gold={'text':'x'*20,'spans':[span('Prazo',0,10,['Explicito']),span('Prazo',0,10,['Implicito'])],
              'relations':[]}
        ann={'text':'x'*20,'spans':[span('Prazo',1,11,['Condicionado'])], 'relations':[]}
        out,r=enrich_version(gold,ann)
        self.assertEqual([(s['begin'],s['end'],s['properties']) for s in out['spans']],
                         [(0,10,['Explicito']),(0,10,['Implicito'])])
        self.assertEqual(r['unmapped_gold_spans'],2)


if __name__=='__main__':unittest.main()
