"""CPU gates for configurable polling without changing flow-control bounds."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pydantic import ValidationError
import yaml

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config, load_config
from specforge.config.schema import RuntimeConfig
from specforge.launch import build_disagg_online_consumer
from specforge.runtime.control_plane.metadata_store import InMemoryMetadataStore
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.training.disaggregated import _build_online


ALGORITHM=builtin_algorithm_registry().resolve('dflash')
POLL_FIELDS=('producer_backpressure_poll_s','consumer_ref_poll_s','ref_distributor_poll_s')
DEFAULTS=(0.2,0.1,0.05)
TUNED=(0.011,0.013,0.017)
BOUNDS={'producer_lease':1,'producer_concurrency':1,
        'in_flight_high_watermark':4,'in_flight_low_watermark':2,
        'resident_high_watermark_bytes':512<<20,'resident_low_watermark_bytes':256<<20,
        'feature_store_max_resident_bytes':1<<30}


def payload(root,role,*,polling=None):
    runtime=dict(BOUNDS)
    if polling is not None:runtime.update(zip(POLL_FIELDS,polling))
    return {'model':{'target_model_path':'target','draft_model_config':'draft'},
        'data':{'prompts_path':'prompts.jsonl','dataloader_num_workers':0},
        'training':{'strategy':'dflash','role':role,'max_steps':1,'num_anchors':8,
                    'seed':42,'prompt_seed':42},'runtime':runtime,
        'deployment':{'mode':'disaggregated','disaggregated':{
            'control_dir':str(root),'backend':'mooncake','async_ack':False,
            'server_urls':['http://capture:30000']}}}


class PollingSchemaTests(unittest.TestCase):
    def test_defaults_preserve_fixed_upstream_cadences_and_bounds(self):
        runtime=RuntimeConfig()
        self.assertEqual(tuple(getattr(runtime,name) for name in POLL_FIELDS),DEFAULTS)
        self.assertEqual((runtime.in_flight_high_watermark,runtime.in_flight_low_watermark),(256,192))
        self.assertEqual(runtime.producer_concurrency,1)

    def test_positive_finite_values_do_not_change_any_other_runtime_field(self):
        default=RuntimeConfig(**BOUNDS).model_dump()
        tuned=RuntimeConfig(**BOUNDS,**dict(zip(POLL_FIELDS,TUNED))).model_dump()
        for field,value in zip(POLL_FIELDS,TUNED):
            self.assertEqual(tuned.pop(field),value)
            default.pop(field)
        self.assertEqual(tuned,default)

    def test_nonpositive_nonfinite_and_non_numeric_values_are_rejected(self):
        for field in POLL_FIELDS:
            for value in (0.,-0.,-0.01,float('nan'),float('inf'),float('-inf'),True,False,'invalid',None):
                with self.subTest(field=field,value=value),self.assertRaises(ValidationError):
                    RuntimeConfig(**{field:value})

    def test_actual_yaml_loader_and_dotted_overrides_revalidate_polling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.yaml'
            path.write_text(yaml.safe_dump(payload(tmp,'consumer',polling=TUNED)))
            cfg=load_config(str(path),['runtime.consumer_ref_poll_s=0.023'])
            self.assertEqual(cfg.runtime.producer_backpressure_poll_s,TUNED[0])
            self.assertEqual(cfg.runtime.consumer_ref_poll_s,0.023)
            self.assertEqual(cfg.runtime.ref_distributor_poll_s,TUNED[2])
            for field in POLL_FIELDS:
                for value in ('0.0','-0.01','nan','inf','-inf','true'):
                    with self.subTest(field=field,value=value),self.assertRaises(ValidationError):
                        load_config(str(path),[f'runtime.{field}={value}'])


class RoleFactoryWiringTests(unittest.TestCase):
    def test_producer_config_reaches_existing_builder_parameter_with_bounds_unchanged(self):
        for polling,expected in ((None,DEFAULTS),(TUNED,TUNED)):
            with self.subTest(polling=polling),tempfile.TemporaryDirectory() as tmp:
                cfg=Config.model_validate(payload(tmp,'producer',polling=polling))
                with (
                    mock.patch.dict(os.environ,{'DISAGG_REF_CHANNEL':str(Path(tmp)/'refs'),
                        'DISAGG_RESIDENT_HIGH_WATERMARK_BYTES':'',
                        'DISAGG_RESIDENT_LOW_WATERMARK_BYTES':''}),
                    mock.patch('specforge.runtime.data_plane.streaming_ref_channel.StreamingRefChannel'),
                    mock.patch('specforge.training.disaggregated._mooncake_store'),
                    mock.patch('specforge.training.assembly._load_input_tools'),
                    mock.patch('specforge.training.model_loading.resolve_draft_config'),
                    mock.patch('specforge.training.disaggregated._producer_capture_metadata',return_value=([1,2],8,32,32)),
                    mock.patch('specforge.inference.adapters.server_capture.SGLangServerCaptureAdapter'),
                    mock.patch('specforge.launch.build_disagg_online_producer',return_value=([],mock.Mock())) as build,
                ):
                    _build_online(cfg,algorithm=ALGORITHM,build_model_bundle=mock.Mock(),
                        prepare_prompts=lambda *a,**k:[{'task_id':'p0','payload':{'input_ids':[1,2,3]}}],
                        optimizer_factory=mock.Mock(),logger=None)
                args=build.call_args.kwargs
                self.assertEqual(args['backpressure_poll_s'],expected[0])
                for name in ('producer_concurrency','in_flight_high_watermark','in_flight_low_watermark',
                             'resident_high_watermark_bytes','resident_low_watermark_bytes',
                             'feature_store_max_resident_bytes'):
                    self.assertEqual(args[name],BOUNDS[name])
                self.assertEqual(args['lease'],1)
                self.assertEqual(args['prompt_seed'],42)

    def test_consumer_config_reaches_builder_without_changing_ack_or_math(self):
        for polling,expected in ((None,DEFAULTS),(TUNED,TUNED)):
            with self.subTest(polling=polling),tempfile.TemporaryDirectory() as tmp:
                cfg=Config.model_validate(payload(tmp,'consumer',polling=polling))
                bundle=SimpleNamespace(model=object(),target_head=None,strategy_kwargs={})
                with (
                    mock.patch.dict(os.environ,{'DISAGG_REF_CHANNEL':str(Path(tmp)/'refs')}),
                    mock.patch('specforge.runtime.data_plane.streaming_ref_channel.StreamingRefChannel'),
                    mock.patch('specforge.training.disaggregated._mooncake_store'),
                    mock.patch('specforge.launch.build_disagg_online_consumer',return_value=SimpleNamespace()) as build,
                ):
                    _build_online(cfg,algorithm=ALGORITHM,build_model_bundle=lambda c:bundle,
                        prepare_prompts=mock.Mock(),optimizer_factory=mock.Mock(),logger=None)
                args=build.call_args.kwargs
                self.assertEqual(args['ref_poll_s'],expected[1])
                self.assertEqual(args['distributor_poll_s'],expected[2])
                self.assertIs(args['draft_model'],bundle.model)
                self.assertEqual(args['async_ack'],False)
                self.assertEqual(args['dispatch_num_anchors'],8)
                self.assertEqual(args['dataloader_num_workers'],0)
                self.assertEqual(args['max_steps'],1)


class EffectiveConsumerPollingTests(unittest.TestCase):
    def build(self,root,**polling):
        channel=StreamingRefChannel(str(root/'refs.jsonl'))
        store=SimpleNamespace(retain_on_release=True)
        captured={}
        def assemble(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()
        with (
            mock.patch('torch.distributed.is_initialized',return_value=False),
            mock.patch.object(RefDistributor,'start',autospec=True,side_effect=lambda self:self) as start,
            mock.patch('specforge.launch._assemble_trainer',side_effect=assemble),
        ):
            trainer=build_disagg_online_consumer(algorithm=ALGORITHM,feature_store=store,channel=channel,
                draft_model=object(),optimizer_factory=object(),run_id='polling-test',output_dir=str(root/'out'),
                collate_fn=lambda features:{},metadata_store=InMemoryMetadataStore(),
                inbox_dir=str(root/'inboxes'),async_ack=False,idle_timeout_s=12.5,**polling)
        self.assertEqual(start.call_count,1)
        return trainer,captured

    def test_existing_direct_call_keeps_real_queue_and_distributor_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer,captured=self.build(Path(tmp))
        self.assertEqual(captured['ref_source']['queue'].poll_s,0.1)
        self.assertEqual(trainer.ref_distributor.poll_s,0.05)
        self.assertEqual(captured['ref_source']['queue'].idle_timeout_s,12.5)
        self.assertEqual(trainer.ref_distributor.idle_timeout_s,12.5)

    def test_custom_builder_values_reach_real_constructor_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer,captured=self.build(Path(tmp),ref_poll_s=TUNED[1],distributor_poll_s=TUNED[2])
        self.assertEqual(captured['ref_source']['queue'].poll_s,TUNED[1])
        self.assertEqual(trainer.ref_distributor.poll_s,TUNED[2])
        self.assertTrue(captured['ref_source']['defer_ack_until_durable'])
        self.assertEqual(captured['async_ack'],False)
        self.assertEqual(trainer.ref_distributor.refs_per_rank_step,1)


if __name__=='__main__':unittest.main()
