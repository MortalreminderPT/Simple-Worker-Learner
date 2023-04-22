
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13policy_buffer.proto\x12\rpolicy_buffer\"F\n\x06Policy\x12\x13\n\x0bglobal_step\x18\x01 \x01(\x05\x12\x12\n\nstate_dict\x18\x02 \x01(\x0c\x12\x13\n\x0bnum_workers\x18\x03 \x01(\x05\"\xa7\x01\n\rRollingResult\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x11\n\tworker_id\x18\x02 \x01(\x05\x12\x0e\n\x06\x62uffer\x18\x03 \x01(\x0c\x12\x38\n\x06reward\x18\x04 \x03(\x0b\x32(.policy_buffer.RollingResult.RewardEntry\x1a-\n\x0bRewardEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x32\x93\x01\n\x06Server\x12@\n\tGetPolicy\x12\x1c.policy_buffer.RollingResult\x1a\x15.policy_buffer.Policy\x12G\n\x10RollingAndUpdate\x12\x1c.policy_buffer.RollingResult\x1a\x15.policy_buffer.Policyb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'policy_buffer_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _ROLLINGRESULT_REWARDENTRY._options = None
  _ROLLINGRESULT_REWARDENTRY._serialized_options = b'8\001'
  _POLICY._serialized_start=38
  _POLICY._serialized_end=108
  _ROLLINGRESULT._serialized_start=111
  _ROLLINGRESULT._serialized_end=278
  _ROLLINGRESULT_REWARDENTRY._serialized_start=233
  _ROLLINGRESULT_REWARDENTRY._serialized_end=278
  _SERVER._serialized_start=281
  _SERVER._serialized_end=428
