#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import typing
import random
import struct
import sys
import uuid

import psf
import upgen_v1
from upgen_v1 import HandshakePattern, KeyPattern, set_parameters

sys.path.insert(0, './greeting')

from predict import load_model, predict
from train import Params as Params

def cipher_of_security_param(secparam: int) -> psf.Cipher:
    match secparam:
        case 128:
            return psf.Cipher.AES128GCM
        case 256:
            return random.choice((psf.Cipher.AES256GCM, psf.Cipher.CHACHA20POLY1305))
        case _:
            raise NotImplementedError


def formats_of_handshake_pattern(
    handshake_pattern: HandshakePattern,
    greeting: bool,
    reversed_greeting: bool,
    subprotocol_nrounds,
) -> list[psf.Format]:
    n = len(handshake_pattern.key_patterns)

    format_id_strs = [f"handshake{i+1}" for i in range(0, n)]

    if greeting:
        if reversed_greeting:
            format_id_strs.insert(0, "greeting_client")
            format_id_strs.insert(0, "greeting_server")
        else:
            format_id_strs.insert(0, "greeting_server")
            format_id_strs.insert(0, "greeting_client")

    for idx in range(subprotocol_nrounds):
        format_id_strs.append(f"handshake_subprotocol_client_{idx}")
        format_id_strs.append(f"handshake_subprotocol_server_{idx}")

    format_id_strs.append("data")

    return [psf.Format(psf.Identifier(s), fields=[]) for s in format_id_strs]


def formats_id_strs_with_payload_of_handshake(
    handshake_pattern: HandshakePattern,
) -> list[str]:
    retval = []

    idx = 0

    for round in handshake_pattern.key_patterns:
        idx += 1

        for key in round:
            match key:
                case KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA:
                    retval.append(f"handshake{idx}")

    return retval + ["data"]


def instantiate_format_fields(
    formats: list[psf.Format],
    fields,
    field_order,
    handshake_pattern: HandshakePattern,
    cipher: psf.Cipher,
    key_encoding: psf.PubkeyEncoding,
    subprotocol_sizes: list[int],
    send_encrypted_cert: bool,
    encrypted_cert_nbytes: int,
):
    format_of_format_id_str = {str(f.name): f for f in formats}

    def append_field_to_formats(format_id_strs: list[str], field: psf.Field):
        formats = [format_of_format_id_str[s] for s in format_id_strs]
        for format in formats:
            format.fields.append(field)

    handshake_format_id_strs = [
        str(f.name) for f in formats if f.name.startswith("handshake")
    ]

    # Add greetings if needed.
    for format in formats:
        if format.name == "greeting_client" or format.name == "greeting_server":
            greeting_field = psf.Field(
                psf.Identifier("greeting"),
                psf.PrimitiveArray(psf.NumericType.U8, 0),
            )
            append_field_to_formats([str(format.name)], greeting_field)

    # Start with unencrypted fields..
    assert all([f in field_order for f in ["unencrypted", "encrypted"]])
    for field in (
        field_order["unencrypted"]
        + ["__end_unencrypted"]
        + field_order["encrypted"]
        + ["__end_encrypted"]
    ):
        match field:
            case "extra":
                extra_handshake_nbytes = fields["extra"]["handshake_length"]
                extra_data_nbytes = fields["extra"]["data_length"]

                if extra_handshake_nbytes > 0:
                    extra_handshake_field = psf.Field(
                        psf.Identifier("extra"),
                        psf.PrimitiveArray(psf.NumericType.U8, extra_handshake_nbytes),
                    )
                    append_field_to_formats(
                        handshake_format_id_strs, extra_handshake_field
                    )

                if extra_data_nbytes > 0:
                    extra_data_field = psf.Field(
                        psf.Identifier("extra"),
                        psf.PrimitiveArray(psf.NumericType.U8, extra_data_nbytes),
                    )
                    append_field_to_formats(["data"], extra_data_field)

            case "length":
                length_nbytes = fields["length"]["length"]

                if length_nbytes == 2:
                    length_type = psf.NumericType.U16
                elif length_nbytes == 4:
                    length_type = psf.NumericType.U32
                else:
                    raise NotImplementedError

                length_field = psf.Field(
                    name=psf.Identifier("length"), type_value=length_type
                )

                append_field_to_formats(
                    handshake_format_id_strs + ["data"], length_field
                )
            case "padding_length":
                # Never pad if CHACHA
                if cipher == psf.Cipher.CHACHA20POLY1305:
                    continue

                # This is the padding LENGTH field
                padding_length_nbytes = fields["padding_length"]["length"]

                if padding_length_nbytes == 2:
                    padding_length_type = psf.NumericType.U16
                elif padding_length_nbytes == 4:
                    padding_length_type = psf.NumericType.U32
                else:
                    raise NotImplementedError

                padding_length_field = psf.Field(
                    name=psf.Identifier("padding_length"),
                    type_value=padding_length_type,
                )

                append_field_to_formats(
                    formats_id_strs_with_payload_of_handshake(handshake_pattern),
                    padding_length_field,
                )
            case "nonce":
                if fields["nonce"]["exists"]:
                    print(type(cipher))
                    randomness_nbytes = cipher.iv_length_nbytes()

                    nonce_field = psf.Field(
                        name=psf.Identifier("nonce"),
                        type_value=psf.PrimitiveArray(
                            psf.NumericType.U8, randomness_nbytes
                        ),
                    )

                    append_field_to_formats(["handshake1"], nonce_field)

                    if "handshake2" in handshake_format_id_strs:
                        append_field_to_formats(["handshake2"], nonce_field)
            case "reserved":
                reserved_nbytes = fields["reserved"]["length"]

                reserved_field = psf.Field(
                    name=psf.Identifier("reserved"),
                    type_value=psf.PrimitiveArray(psf.NumericType.U8, reserved_nbytes),
                )

                if reserved_nbytes > 0:
                    append_field_to_formats(handshake_format_id_strs, reserved_field)
            case "type":
                # Type currently goes in all PDUs, always
                type_field = psf.Field(
                    name=psf.Identifier("type"),
                    type_value=psf.PrimitiveArray(psf.NumericType.U8, 1),
                )
                append_field_to_formats(handshake_format_id_strs + ["data"], type_field)
            case "version":
                version_nbytes = fields["version"]["length"]

                version_field = psf.Field(
                    name=psf.Identifier("version"),
                    type_value=psf.PrimitiveArray(psf.NumericType.U8, version_nbytes),
                )

                if fields["version"]["in_handshake"]:
                    append_field_to_formats(["handshake1"], version_field)

                    if "handshake2" in handshake_format_id_strs:
                        append_field_to_formats(["handshake2"], version_field)

            case "__end_unencrypted":
                pass
            case "__end_encrypted":

                def has_encrypted_headers(format: psf.Format) -> bool:
                    for field in format.fields:
                        if str(field.name) in field_order["encrypted"]:
                            return True
                    return False

                if field_order["overall_structure"] == 0:
                    # place a MAC at the end of the encrypted headers

                    header_mac_field = psf.Field(
                        psf.Identifier("header_mac"),
                        psf.PrimitiveArray(psf.NumericType.U8, cipher.mac_tag_nbytes()),
                    )

                    for format in formats:
                        if has_encrypted_headers(format):
                            append_field_to_formats(
                                [str(format.name)], header_mac_field
                            )
                else:
                    assert field_order["overall_structure"] == 1

                # Attach any needed key fields
                for format_id_str, key_pattern in zip(
                    handshake_format_id_strs, handshake_pattern.key_patterns
                ):
                    for idx, key in enumerate(key_pattern):
                        match key:
                            case (
                                KeyPattern.EPHEMERAL
                                | KeyPattern.EPHEMERAL_WITH_OPTIONAL_DATA
                            ):
                                id_str = "ephemeral_key"
                            case KeyPattern.STATIC | KeyPattern.ENCRYPTED_STATIC:
                                id_str = "static_key"

                        print(key)

                        key_field = psf.Field(
                            psf.Identifier(id_str),
                            psf.PrimitiveArray(
                                psf.NumericType.U8,
                                key_encoding.encoding_size_nbytes(),
                            ),
                        )

                        append_field_to_formats([format_id_str], key_field)

                payload_field = psf.Field(
                    psf.Identifier("payload"),
                    psf.DynamicArray(psf.Identifier("length")),
                )

                padding_field = psf.Field(
                    psf.Identifier("padding"),
                    psf.DynamicArray(psf.Identifier("padding_length")),
                )

                mac_field = psf.Field(
                    psf.Identifier("msg_mac"),
                    psf.PrimitiveArray(psf.NumericType.U8, cipher.mac_tag_nbytes()),
                )

                fids_with_data = formats_id_strs_with_payload_of_handshake(
                    handshake_pattern
                )

                for format in formats:
                    has_data = str(format.name) in fids_with_data

                    if has_data:
                        append_field_to_formats([str(format.name)], payload_field)

                        if cipher != psf.Cipher.CHACHA20POLY1305:
                            append_field_to_formats([str(format.name)], padding_field)

                    if (
                        has_encrypted_headers(format)
                        and field_order["overall_structure"] == 1
                    ) or has_data:
                        append_field_to_formats([str(format.name)], mac_field)

            case s:
                logging.error(f"Received unexpected field: {s}")
                raise NotImplementedError

    # Now, we need to prune out length fields. If a format has no variable lenght
    # fields, it should not have a length field.
    def has_dynamic_field(format: psf.Format) -> bool:
        return any(type(f.type_value) == psf.DynamicArray for f in format.fields)

    for format in formats:
        if not has_dynamic_field(format):
            format.fields = [f for f in format.fields if str(f.name) != "length"]

    # And finally, we hack in subprotocol handshakes. The subprotocol packets should
    # appear like ordinary data packets, so we steal them.

    data_format = copy.deepcopy(next(filter(lambda f: f.name == "data", formats)))

    new_data_fields: list[psf.Field] = []
    for field in data_format.fields:
        if field.name == "payload":
            new_field = copy.deepcopy(field)
            new_field.name = "payload_fake"
            new_field.type_value = psf.PrimitiveArray(psf.NumericType.U8, -1)
            new_data_fields.append(new_field)
        elif field.name == "padding_length":
            new_field = copy.deepcopy(field)
            new_field.name = "padding_length_fake"
            new_field.type_value = psf.PrimitiveArray(
                psf.NumericType.U8, field.type_value.repr_nbytes()
            )
            new_data_fields.append(new_field)
        elif field.name == "padding":
            pass
        else:
            new_data_fields.append(field)

    data_format.fields = new_data_fields

    def subproto_length_of_payload(name: str) -> int:
        fs = name.split("_")
        role = fs[-2]
        round = int(fs[-1])

        offset = None

        if role == "client":
            offset = 0
        else:
            assert role == "server"
            offset = 1

        idx = round * 2 + offset

        return subprotocol_sizes[idx]

    def padding_nbytes(payload_nbytes: int) -> int:
        block_size_nbytes = cipher.block_size_nbytes()

        if block_size_nbytes is None:
            return 0
        else:
            rem = payload_nbytes % block_size_nbytes
            if rem == 0:
                return 0
            else:
                delta = block_size_nbytes - rem
                return delta

    if send_encrypted_cert:
        for format in formats:
            if format.name == "handshake2":
                nonce_field = psf.Field(
                    name=psf.Identifier("payload_fake"),
                    type_value=psf.PrimitiveArray(
                        psf.NumericType.U8, encrypted_cert_nbytes + padding_nbytes(encrypted_cert_nbytes)
                    ),
                )

                append_field_to_formats(["handshake2"], nonce_field)

    for format in formats:
        if format.name.startswith("handshake_subprotocol"):
            format.fields = copy.deepcopy(data_format.fields)
            # Now we need to set the lengths correctly.
            for field in format.fields:
                if (
                    field.name == "payload_fake"
                    and field.type_value == psf.PrimitiveArray(psf.NumericType.U8, -1)
                ):
                    lp = subproto_length_of_payload(format.name)
                    lp += padding_nbytes(lp)
                    field.type_value = psf.PrimitiveArray(psf.NumericType.U8, lp)

    # Finally, take a pass to determine the right length field.
    for format in formats:
        if format.name.startswith("handshake_subprotocol"):
            assert lp is not None

            for field in format.fields:
                if field.name == "length":
                    lp = subproto_length_of_payload(format.name)
                    lp += padding_nbytes(lp)
                    field.name = f"length_{lp + cipher.mac_tag_nbytes()}"
                    field.type_value = psf.PrimitiveArray(
                        psf.NumericType.U8, field.type_value.repr_nbytes()
                    )


def semantics_of_formats(
    formats: list[psf.Format],
    protocol_settings: upgen_v1.ProtocolSettings,
    gen_greeting_fn: typing.Callable[[], str],
) -> list[psf.SemanticBinding]:
    retval: list[psf.SemanticBinding] = []

    cipher = protocol_settings.cipher
    greeting_str = gen_greeting_fn()

    for format in formats:
        for field in format.fields:
            field_semantic: typing.Optional[psf.FieldSemantic] = None

            match str(field.name):
                case "extra":
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )
                    nbytes = field.type_value.nelems
                    field_semantic = psf.RandomSemantic(nbytes)
                case "length":
                    field_semantic = psf.SemanticValue.LENGTH
                case "padding_length":
                    field_semantic = psf.SemanticValue.PADDING_LENGTH
                case "nonce":
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )
                    nbytes = field.type_value.nelems
                    field_semantic = psf.RandomSemantic(nbytes)
                case "reserved":
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )
                    nbytes = field.type_value.nelems
                    zero_str = "".join(["00" for _ in range(nbytes)])
                    field_semantic = psf.FixedBytesSemantic(
                        value=psf.HexLiteral(f"0x{zero_str}")
                    )
                case "padding":
                    field_semantic = psf.SemanticValue.PADDING
                case "payload":
                    field_semantic = psf.SemanticValue.PAYLOAD
                case "type":
                    is_handshake = str(format.name).startswith("handshake")

                    # The types are
                    # Normal:
                    # 1. Handshake
                    # 2. Data
                    # 3. Session close
                    # 4. Heartbeat
                    # Control:
                    # 5. Alert
                    # 6. Debug

                    if is_handshake:
                        type_offset = 0
                    else:
                        assert str(format.name) == "data"
                        type_offset = 1

                    if not protocol_settings.fields["type"]["normal_types_first"]:
                        type_offset += (
                            protocol_settings.fields["type"]["num_type_nums"] - 4
                        )

                    type_value = (
                        protocol_settings.fields["type"]["start_num"] + type_offset
                    )

                    field_semantic = psf.FixedBytesSemantic(
                        value=psf.HexLiteral(f"0x{type_value:02X}")
                    )

                case "version":
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )

                    version_settings = protocol_settings.fields["version"]

                    assert version_settings["length"] in (1, 2)

                    value_str = f"0x{version_settings['major_val']:02X}"

                    if version_settings["length"] == 2:
                        value_str += f"{version_settings['minor_val']:02X}"

                    field_semantic = psf.FixedBytesSemantic(
                        value=psf.HexLiteral(value_str)
                    )

                case "header_mac" | "msg_mac":
                    field_semantic = psf.RandomSemantic(cipher.mac_tag_nbytes())
                case "ephemeral_key" | "static_key":
                    field_semantic = psf.PubkeySemantic(
                        psf.PubkeyEncoding(
                            psf.PubkeyEncoding.of_str(protocol_settings.key_encoding)
                        )
                    )
                case "greeting":
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )

                    field.type_value = psf.PrimitiveArray(
                        psf.NumericType.U8, len(greeting_str)
                    )

                    field_semantic = psf.FixedStringSemantic(value=greeting_str)
                case "padding_length_fake" | "payload_fake":
                    # Same logic as nonce.
                    assert (
                        type(field.type_value) == psf.PrimitiveArray
                        and field.type_value.primitive_type == psf.NumericType.U8
                    )
                    nbytes = field.type_value.nelems
                    field_semantic = psf.RandomSemantic(nbytes)

                case s:
                    if s.startswith("length_"):
                        length_field_length = field.type_value.nelems

                        if length_field_length == 2:
                            spec = ">H"
                        elif length_field_length == 4:
                            spec = ">I"
                        else:
                            raise NotImplementedError

                        length_nbytes = int(s.split("_")[1])
                        length_bytes = struct.pack(spec, length_nbytes)
                        field_semantic = psf.FixedBytesSemantic(
                            value=psf.HexLiteral("0x" + length_bytes.hex())
                        )
                    else:
                        logging.error(f"Received unexpected field: {s}")
                        raise NotImplementedError

            if field_semantic is not None:
                retval.append(
                    psf.SemanticBinding(
                        format_id=format.name,
                        field_id=field.name,
                        semantic=field_semantic,
                    )
                )

    return retval


def sequence_specifier_of_formats(
    formats: list[psf.Format],
) -> list[psf.SequenceSpecifier]:
    retval = []

    for format in formats:
        name = str(format.name)
        if name == "greeting_client":
            retval.append(
                psf.SequenceSpecifier(
                    role=psf.Role.CLIENT,
                    phase=psf.Phase.HANDSHAKE,
                    format_id=format.name,
                )
            )
        elif name == "greeting_server":
            retval.append(
                psf.SequenceSpecifier(
                    role=psf.Role.SERVER,
                    phase=psf.Phase.HANDSHAKE,
                    format_id=format.name,
                )
            )
        elif name.startswith("handshake_subprotocol"):
            subprotocol_num = int(name.split("_")[-1])
            role = psf.Role.of_str(name.split("_")[-2])

            retval.append(
                psf.SequenceSpecifier(
                    role=role, phase=psf.Phase.HANDSHAKE, format_id=format.name
                )
            )
        elif name.startswith("handshake"):
            msg_num = int(name.split("handshake")[-1])
            role = psf.Role.CLIENT if msg_num % 2 == 1 else psf.Role.SERVER
            retval.append(
                psf.SequenceSpecifier(
                    role=role, phase=psf.Phase.HANDSHAKE, format_id=format.name
                )
            )
        else:
            assert name == "data"

            for role in (psf.Role.CLIENT, psf.Role.SERVER):
                retval.append(
                    psf.SequenceSpecifier(
                        role=role, phase=psf.Phase.DATA, format_id=format.name
                    )
                )

    return retval


def make_encryption_directives(
    formats: list[psf.Format],
    protocol_settings: upgen_v1.ProtocolSettings,
) -> list[psf.EncryptionDirectives]:
    retval: list[psf.EncryptionDirectives] = []

    def encrypted(format_id_str, field_id_str):
        if (
            field_id_str == "padding_length"
            or field_id_str == "payload"
            or field_id_str == "padding"
        ):
            return True

        if format_id_str == "handshake3" and field_id_str == "static_key":
            return True

        if field_id_str in protocol_settings.field_order["encrypted"]:
            return True

        return False

    for format in formats:
        efds = []

        for field in format.fields:
            if encrypted(format.name, field.name):
                efds.append(psf.EncryptionFieldDirective(field.name, field.name, None))

        if len(efds) > 0:
            efb = psf.EncryptionFormatBinding(format.name, format.name)
            retval.append(psf.EncryptionDirectives(efb, efds))

    return retval


def generate_psf(
    protocol_settings: upgen_v1.ProtocolSettings,
    gen_greeting_fn: typing.Callable[[], str],
) -> psf.ProtocolSpecificationFile:
    logging.info(f"Sampled protocol settings:\n{protocol_settings}")

    cipher = protocol_settings.cipher

    key_encoding = psf.PubkeyEncoding.of_str(protocol_settings.key_encoding)

    formats = formats_of_handshake_pattern(
        protocol_settings.handshake_pattern,
        protocol_settings.greeting,
        protocol_settings.reversed_greeting,
        protocol_settings.subprotocol_nrounds,
    )

    instantiate_format_fields(
        formats,
        protocol_settings.fields,
        protocol_settings.field_order,
        protocol_settings.handshake_pattern,
        cipher,
        key_encoding,
        protocol_settings.subprotocol_sizes,
        protocol_settings.send_encrypted_cert,
        protocol_settings.encrypted_cert_nbytes,
    )

    semantics = semantics_of_formats(formats, protocol_settings, gen_greeting_fn)
    sequence = sequence_specifier_of_formats(formats)

    crypto = psf.CryptoSegment(
        pass_assn=psf.PasswordAssignment(str(uuid.uuid4())),
        cipher_assn=psf.CipherAssignment(cipher),
        enc_dirs=make_encryption_directives(formats, protocol_settings),
    )

    options = psf.OptionsSegment(
        separate_length_field=psf.SeparateLengthFieldSetting(protocol_settings.separate_length_fields)
    )

    p = psf.ProtocolSpecificationFile(
        formats=formats,
        semantics=semantics,
        sequence=sequence,
        crypto=crypto,
        options=options,
    )

    return p


def main(args: argparse.Namespace):
    log_level_of_str = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    format = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
    logging.basicConfig(
        level=log_level_of_str[args.log_level], format=format, stream=sys.stderr
    )
    logging.info(f"Program arguments:\t{args}")

    with open(args.config_filepath, 'r') as in_f:
        config = json.load(in_f)

    set_parameters(config["parameters"])

    if args.seed is not None:
        random.seed(args.seed)

    if args.output_filepath == "-":
        out_f = sys.stdout
    else:
        out_f = open(args.output_filepath, "w")

    model, encoder, params, dev = load_model(
        args.model_filepath, args.encoder_filepath, args.best_params_filepath, None
    )

    def gen_greeting():
        return predict(model, encoder, params, dev, 1, args.greeting_string_temp)[0]

    if args.best:
        sample = upgen_v1.best_protocol_settings
    elif args.worst:
        sample = upgen_v1.worst_protocol_settings

        def gen_greeting():
            return "he"*50

    else:
        sample = upgen_v1.sample_protocol_settings

    for idx in range(args.num_generated):

        logging.info(f"Completed {idx+1} PSFs.")

        protocol_settings = sample()
        p = generate_psf(protocol_settings, gen_greeting)
        print(p, file=out_f)

        if args.num_generated > 1:
            print(bytes.fromhex("1E").decode(), file=out_f)

    out_f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-o", "--output_filepath", type=str, default="-")
    parser.add_argument("-t", "--greeting_string_temp", type=float, default=0.9)
    parser.add_argument("-n", "--num_generated", type=int, default=1)

    parser.add_argument("-b", "--best", action="store_true")
    parser.add_argument("-w", "--worst", action="store_true")

    parser.add_argument("config_filepath")
    parser.add_argument("best_params_filepath")
    parser.add_argument("encoder_filepath")
    parser.add_argument("model_filepath")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
