
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import rpc.policy_buffer_pb2 as policy__buffer__pb2
class ServerStub(object):
    """Missing associated documentation comment in .proto file."""
    def __init__(self, channel):
        """Constructor.
        Args:
            channel: A grpc.Channel.
        """
        self.GetPolicy = channel.unary_unary(
                '/policy_buffer.Server/GetPolicy',
                request_serializer=policy__buffer__pb2.RollingResult.SerializeToString,
                response_deserializer=policy__buffer__pb2.Policy.FromString,
                )
        self.RollingAndUpdate = channel.unary_unary(
                '/policy_buffer.Server/RollingAndUpdate',
                request_serializer=policy__buffer__pb2.RollingResult.SerializeToString,
                response_deserializer=policy__buffer__pb2.Policy.FromString,
                )
class ServerServicer(object):
    """Missing associated documentation comment in .proto file."""
    def GetPolicy(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
    def RollingAndUpdate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
def add_ServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetPolicy': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPolicy,
                    request_deserializer=policy__buffer__pb2.RollingResult.FromString,
                    response_serializer=policy__buffer__pb2.Policy.SerializeToString,
            ),
            'RollingAndUpdate': grpc.unary_unary_rpc_method_handler(
                    servicer.RollingAndUpdate,
                    request_deserializer=policy__buffer__pb2.RollingResult.FromString,
                    response_serializer=policy__buffer__pb2.Policy.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'policy_buffer.Server', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
 
class Server(object):
    """Missing associated documentation comment in .proto file."""
    @staticmethod
    def GetPolicy(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/policy_buffer.Server/GetPolicy',
            policy__buffer__pb2.RollingResult.SerializeToString,
            policy__buffer__pb2.Policy.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
    @staticmethod
    def RollingAndUpdate(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/policy_buffer.Server/RollingAndUpdate',
            policy__buffer__pb2.RollingResult.SerializeToString,
            policy__buffer__pb2.Policy.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
