import sys
import selectors
import json
import io
import struct
import SERdemo1Learner.RLbrain_v1, SERdemo1Learner.Learner_v1

request_search = {
    "morpheus": "Follow the white rabbit. \U0001f430",
    "ring": "In the caves beneath the Misty Mountains. \U0001f48d",
    "\U0001f436": "\U0001f43e Playing ball! \U0001f3d0",
}


class Message:
    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self.request = None
        self.response_created = False
        self.content_length = None
        self.content_type = None

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def _write(self):
        if self._send_buffer:
            print("sending", repr(self._send_buffer), "to", self.addr)
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]
                # Close when the buffer is drained. The response has been sent.
                if sent and not self._send_buffer:
                    self.close()

    def _create_message(
        self, *, content_bytes, content_type, content_encoding
    ):
        message_len_hdr = struct.pack(">H", len(content_bytes) )
        message_type_hdr = struct.pack(">B", content_type)
        message = message_len_hdr + message_type_hdr + content_bytes
        return message

    def _create_response_binary_content(self):
        response = {
            "content_bytes": bytes([9,8,7,6,5,4,3,2,1]), #TODO: insert new parameter data here
            "content_type": 1,
            "content_encoding": "binary",
        }
        return response

    def process_events(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        self._read()

        self.process_protoheader()

        self.process_request()

    def write(self):
        if self.request:
            if not self.response_created:
                self.create_response()

        self._write()

    def close(self):
        print("closing connection to", self.addr)
        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            print(
                f"error: selector.unregister() exception for",
                f"{self.addr}: {repr(e)}",
            )

        try:
            self.sock.close()
        except OSError as e:
            print(
                f"error: socket.close() exception for",
                f"{self.addr}: {repr(e)}",
            )
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None

    def process_protoheader(self):
        hdrlen = 3
        self.content_length = struct.unpack(
            ">H", self._recv_buffer[:2]
        )[0]
        self.content_type = struct.unpack(
            ">B", self._recv_buffer[2:3]
        )[0]
        if self.content_type != 1:
            self._recv_buffer = self._recv_buffer[hdrlen:]
        else: print("Pull request")
        
    def create_response(self):
        if self.content_type == 1:
            response = self._create_response_binary_content()
            message = self._create_message(**response)
            self.response_created = True
            self._send_buffer += message
        else: 
            message_len_hdr = struct.pack(">H", len(bytes([])) )
            message_type_hdr = struct.pack(">B", 3)
            message = message_len_hdr + message_type_hdr
            self.response_created = True
            self._send_buffer += message


    def process_request(self):
        content_len = self.content_length
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        if self.content_type == 1:
            print("Recieved Parameter Pull Request")
            self.create_response()
        else:
            print("Recieved new experiences")
            self.request = data
            SERdemo1Learner.Learner_v1.receive_exp(data) #TODO: Correct? 
            print(
                f'received a "{self.content_type}" request from',
                self.addr,
            )
            #New experiences recieved, process them here
        # Set selector to listen for write events, we're done reading.
        self._set_selector_events_mask("w")

