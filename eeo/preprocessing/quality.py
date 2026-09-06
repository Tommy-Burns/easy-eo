"""Decoding the quality layers that ship beside a product's reflectance bands.

Every Sentinel-2 and Landsat product carries a per-pixel quality layer, and it
is the only honest way to know which pixels in a scene are actually a view of
the ground. The layers are not comparable to each other: Sentinel-2's ``SCL``
holds one class number per pixel, where Landsat's ``QA_PIXEL`` packs several
independent flags into the bits of a single integer. This module's job is to
turn either of them into the same thing — a boolean array that is ``True``
where a pixel should be discarded — so that everything downstream can be
written once rather than once per mission.

Deciding *which* pixels count as cloud is a judgement rather than a fact, so
the class sets below are defaults and not truths. They are named constants
precisely so that a user who disagrees can say so in their own code, and so
that a published analysis can state exactly what it masked.

Notes
-----
Class numbers are labels, not quantities: the midpoint of "cloud shadows" (3)
and "vegetation" (4) is not "somewhere between the two", it is "not vegetated"
(5). A quality band must therefore only ever be resampled by nearest
neighbour, which is what :attr:`~eeo.io._bands.BandInfo.kind` marks these
bands for. Their dtype after a load is not significant — stacking an integer
``SCL`` alongside float reflectance bands widens it to float — but the values
themselves must have arrived unmodified.

References
----------
Sentinel-2 class numbers and names follow the Scene Classification table
published by Copernicus at https://sentiwiki.copernicus.eu/web/s2-processing.

Landsat bit assignments follow the USGS Collection 2 Level-2 Science Product
Guides: LSDS-1619 Table 6-2 for Landsat 8-9, and LSDS-1618 Table 5-5 for
Landsat 4-7. The two tables differ, which is the whole reason the Landsat
functions here require a mission number.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from enum import IntEnum

import numpy as np

from eeo.core.exceptions import ValidationError


class SCLClass(IntEnum):
    """The twelve classes of the Sentinel-2 Level-2A scene classification.

    The ``SCL`` band assigns every pixel exactly one of these numbers. Member
    names are Copernicus's own, spelled as they appear in the published Scene
    Classification table, so that code reads the same as the documentation a
    user is checking it against. The members are ordered by class number,
    which is also their value, so a member can be used anywhere the raw number
    can::

        SCLClass.CLOUD_HIGH_PROBABILITY == 9

    Two classes have been renamed by ESA since the mission launched, without
    their numbers changing: class 2 was ``DARK_FEATURES`` ("dark features /
    shadows") before the 2022 processing-baseline change, and class 5 is
    widely documented under its older informal name "bare soil". Both older
    spellings are still accepted by :func:`resolve_scl_classes`.

    Attributes
    ----------
    NO_DATA : int
        0 — outside the granule, or otherwise without a measurement.
    SATURATED_OR_DEFECTIVE : int
        1 — the sensor saturated, or the detector reported a defect.
    CAST_SHADOWS : int
        2 — topographic cast shadows: ground shadowed by terrain rather than
        by cloud. Named ``DARK_FEATURES`` in older products and documentation.
    CLOUD_SHADOWS : int
        3 — shadow cast by cloud, detected from cloud geometry.
    VEGETATION : int
        4 — vegetated surface.
    NOT_VEGETATED : int
        5 — a surface that is not vegetated. The class is defined negatively
        and so conflates bare soil, rock and built surfaces; older
        documentation calls it "bare soil", which is narrower than what it
        actually contains.
    WATER : int
        6 — water.
    UNCLASSIFIED : int
        7 — no class was confidently assigned. Not a synonym for "clear": it
        marks where the classifier could not decide, so it is neither a claim
        that the pixel is ground nor that it is cloud.
    CLOUD_MEDIUM_PROBABILITY : int
        8 — probably cloud.
    CLOUD_HIGH_PROBABILITY : int
        9 — almost certainly cloud.
    THIN_CIRRUS : int
        10 — thin high-altitude ice cloud. It scatters and attenuates rather
        than hiding the ground, so a pixel here holds a real measurement of a
        real surface — but a contaminated one.
    SNOW_ICE : int
        11 — snow or ice. Bright and cold, and routinely confused with cloud
        in both directions.

    Examples
    --------
    >>> SCLClass.CLOUD_SHADOWS
    <SCLClass.CLOUD_SHADOWS: 3>
    >>> int(SCLClass.THIN_CIRRUS)
    10
    >>> SCLClass(9).name
    'CLOUD_HIGH_PROBABILITY'
    """

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CAST_SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


# Spellings ESA has moved away from, plus the singular of a plural class name.
# Accepted because the old names are what most tutorials, papers and existing
# scripts say, and silently failing to resolve "bare_soil" would be a worse
# outcome than accepting it.
_SCL_NAME_ALIASES = {
    "dark_features": SCLClass.CAST_SHADOWS,
    "dark_feature": SCLClass.CAST_SHADOWS,
    "dark_area": SCLClass.CAST_SHADOWS,
    "dark_area_pixels": SCLClass.CAST_SHADOWS,
    "cast_shadow": SCLClass.CAST_SHADOWS,
    "casted_shadows": SCLClass.CAST_SHADOWS,
    "cloud_shadow": SCLClass.CLOUD_SHADOWS,
    "bare_soil": SCLClass.NOT_VEGETATED,
    "bare_soils": SCLClass.NOT_VEGETATED,
    "not_vegetated_soil": SCLClass.NOT_VEGETATED,
    "snow_or_ice": SCLClass.SNOW_ICE,
    "snow": SCLClass.SNOW_ICE,
    "ice": SCLClass.SNOW_ICE,
}

#: The classes Easy-EO treats as cloud by default: cloud shadows (3), both
#: cloud probabilities (8, 9), and thin cirrus (10). Medium probability is
#: included because excluding it keeps a band of half-cloud around every cloud
#: edge, and cirrus because a contaminated measurement is still a wrong one.
#: It is a deliberately cautious set: it will discard some clear ground, which
#: costs pixels, rather than admit cloud, which costs correctness.
SCL_CLOUDY: frozenset[SCLClass] = frozenset(
    {
        SCLClass.CLOUD_SHADOWS,
        SCLClass.CLOUD_MEDIUM_PROBABILITY,
        SCLClass.CLOUD_HIGH_PROBABILITY,
        SCLClass.THIN_CIRRUS,
    }
)

#: The classes that carry no usable measurement at all — no data (0) and
#: saturated or defective (1) — whatever the sky was doing. Masking these is
#: not a cloud judgement and there is no sensible reason to keep them.
SCL_NODATA: frozenset[SCLClass] = frozenset(
    {
        SCLClass.NO_DATA,
        SCLClass.SATURATED_OR_DEFECTIVE,
    }
)

#: What :func:`scl_mask` masks when not told otherwise: :data:`SCL_CLOUDY`
#: together with :data:`SCL_NODATA`.
SCL_DEFAULT_MASKED: frozenset[SCLClass] = SCL_CLOUDY | SCL_NODATA


def _resolve_scl_class(value: SCLClass | int | str) -> SCLClass:
    """Resolve one class number or class name to an :class:`SCLClass`."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in SCLClass.__members__:
            return SCLClass[name]
        alias = _SCL_NAME_ALIASES.get(name.lower())
        if alias is not None:
            return alias
        raise ValidationError(
            f"no scene class named {value!r}; the names are "
            f"{', '.join(member.name.lower() for member in SCLClass)}"
        )
    # operator.index rather than isinstance(value, int): a class number read
    # back out of an SCL array is a numpy integer, which is not an int. bool is
    # excluded by hand because it is one, and True is not class 1.
    number = None
    if not isinstance(value, bool):
        try:
            number = operator.index(value)
        except TypeError:
            number = None
    if number is None:
        raise ValidationError(
            f"a scene class must be an SCLClass, a class number 0-11, or a class "
            f"name such as 'cloud_shadows'; got {value!r}"
        )
    try:
        return SCLClass(number)
    except ValueError:
        raise ValidationError(
            f"{number} is not a Sentinel-2 scene class; the classes are numbered "
            f"0-11 ({', '.join(f'{m.value} {m.name.lower()}' for m in SCLClass)})"
        ) from None


def resolve_scl_classes(classes: Iterable[SCLClass | int | str]) -> frozenset[SCLClass]:
    """Resolve an iterable of class numbers or names to :class:`SCLClass` members.

    Parameters
    ----------
    classes : iterable of SCLClass or int or str
        Scene classes, each given as a member, as its number (0-11), or as its
        name, matched case-insensitively (``"cloud_shadows"``). The names ESA
        has renamed are accepted under their older spellings too, so
        ``"bare_soil"`` resolves to :attr:`SCLClass.NOT_VEGETATED` and
        ``"dark_features"`` to :attr:`SCLClass.CAST_SHADOWS`.

    Returns
    -------
    frozenset of SCLClass
        The resolved classes. Duplicates collapse, since naming a class twice
        masks the same pixels once.

    Raises
    ------
    ValidationError
        If ``classes`` is a bare string rather than a sequence of them, if it
        is empty, or if any entry names no scene class. A bare string is
        rejected because iterating one yields characters, so ``"water"`` would
        otherwise be read as five unrelated class names.

    Examples
    --------
    >>> sorted(resolve_scl_classes([3, "thin_cirrus"]))
    [<SCLClass.CLOUD_SHADOWS: 3>, <SCLClass.THIN_CIRRUS: 10>]
    >>> resolve_scl_classes(["bare_soil"])
    frozenset({<SCLClass.NOT_VEGETATED: 5>})
    """
    if isinstance(classes, str):
        raise ValidationError(
            f"expected a sequence of scene classes, such as [3, 8, 9, 10] or "
            f"['cloud_shadows', 'thin_cirrus']; got {classes!r}"
        )
    resolved = frozenset(_resolve_scl_class(value) for value in classes)
    if not resolved:
        raise ValidationError(
            "name at least one scene class to mask; got an empty sequence. To mask "
            "nothing, skip the masking step rather than asking for an empty mask"
        )
    return resolved


def scl_mask(
    scl: np.ndarray, *, classes: Iterable[SCLClass | int | str] | None = None
) -> np.ndarray:
    """Flag the Sentinel-2 pixels belonging to a set of scene classes.

    Reads the ``SCL`` band as what it is — one class number per pixel — and
    reports which pixels fall in the classes being masked. It makes no
    judgement of its own beyond the default class set, and it does not touch
    reflectance data; turning this flag into nodata is the masking operation's
    job.

    Parameters
    ----------
    scl : numpy.ndarray
        The ``SCL`` band, holding one class number per pixel. Any shape and
        any numeric dtype, but the values must have arrived by nearest
        neighbour: an interpolated class number denotes a different class, or
        no class at all.
    classes : iterable of SCLClass or int or str, optional
        Which classes to flag, as members, numbers (0-11), or names. Defaults
        to :data:`SCL_DEFAULT_MASKED` — cloud, cloud shadows, cirrus, and the
        two classes that hold no measurement.

    Returns
    -------
    numpy.ndarray
        A boolean array of ``scl``'s shape, ``True`` where the pixel belongs
        to one of ``classes``. Carries no nodata of its own: a flag is either
        set or it is not, and ``SCL``'s own missing-data class is a class like
        any other.

    Raises
    ------
    ValidationError
        If ``classes`` is a bare string, is empty, or names no scene class.

    Notes
    -----
    Elementwise and array-agnostic: the result is backed by whatever backs
    ``scl``, so a lazy array yields a lazy mask and nothing is materialized
    here.

    Examples
    --------
    >>> import numpy as np
    >>> scl = np.array([[4, 9], [6, 3]], dtype="uint8")
    >>> scl_mask(scl)
    array([[False,  True],
           [False,  True]])
    >>> scl_mask(scl, classes=["water"])
    array([[False, False],
           [ True, False]])
    """
    wanted = SCL_DEFAULT_MASKED if classes is None else resolve_scl_classes(classes)
    # A plain array of the class numbers: np.isin does not accept a set, whose
    # iteration order is unspecified, and sorting keeps the call reproducible.
    values = np.array(sorted(int(member) for member in wanted), dtype="int16")
    return np.isin(scl, values)


class QAPixelFlag(IntEnum):
    """The single-bit flags of the Landsat Collection 2 ``QA_PIXEL`` band.

    Each member's value is its **bit position**, not a mask: ``CLOUD`` is 3,
    meaning bit 3, which is the value 8. Bit positions are what the USGS
    tables list, so this keeps the code readable against the documentation.

    Every flag is a claim about *high* confidence, not about presence: bit 3
    is set where cloud confidence is high, and clear where it is merely
    medium. A scene therefore has cloud that no flag marks, which is why
    :func:`qa_pixel_confidence` exists beside these.

    Attributes
    ----------
    FILL : int
        Bit 0 — no image data. Outside the scene's footprint.
    DILATED_CLOUD : int
        Bit 1 — a buffer grown around detected cloud, catching the edge
        pixels that are partly cloud and which the cloud flag itself misses.
    CIRRUS : int
        Bit 2 — high-confidence cirrus. **Landsat 8 and 9 only**: the bit is
        Unused on Landsat 4, 5 and 7, whose sensors have no cirrus band.
    CLOUD : int
        Bit 3 — high-confidence cloud.
    CLOUD_SHADOW : int
        Bit 4 — high-confidence cloud shadow.
    SNOW : int
        Bit 5 — high-confidence snow or ice cover.
    WATER : int
        Bit 7 — water rather than land or cloud.
    CLEAR : int
        Bit 6 — neither the cloud nor the dilated-cloud bit is set. It is
        derived from those two and adds no information of its own, so masking
        on ``not CLEAR`` and masking on ``CLOUD | DILATED_CLOUD`` are the same
        operation.

    Examples
    --------
    >>> QAPixelFlag.CLOUD
    <QAPixelFlag.CLOUD: 3>
    >>> 1 << QAPixelFlag.CLOUD  # the bit's value
    8
    """

    FILL = 0
    DILATED_CLOUD = 1
    CIRRUS = 2
    CLOUD = 3
    CLOUD_SHADOW = 4
    SNOW = 5
    CLEAR = 6
    WATER = 7


class QAConfidenceField(IntEnum):
    """The paired-bit confidence fields of ``QA_PIXEL``.

    Each member's value is the **low bit** of its two-bit field, so
    ``CLOUD`` is 8, meaning bits 8-9. The fields are what USGS validates the
    band against — the single-bit flags above are derived from them — so a
    mask built on confidence is the more defensible one.

    Attributes
    ----------
    CLOUD : int
        Bits 8-9. The only field with a Medium level; see
        :class:`QAConfidence`.
    CLOUD_SHADOW : int
        Bits 10-11.
    SNOW_ICE : int
        Bits 12-13.
    CIRRUS : int
        Bits 14-15. **Landsat 8 and 9 only**; Unused on Landsat 4, 5 and 7.

    Examples
    --------
    >>> QAConfidenceField.SNOW_ICE
    <QAConfidenceField.SNOW_ICE: 12>
    """

    CLOUD = 8
    CLOUD_SHADOW = 10
    SNOW_ICE = 12
    CIRRUS = 14


class QAConfidence(IntEnum):
    """The four values a ``QA_PIXEL`` confidence field can take.

    Attributes
    ----------
    NONE : int
        0 — no confidence level set.
    LOW : int
        1 — low confidence.
    MEDIUM : int
        2 — medium confidence, **for cloud confidence only**. USGS documents
        this value as Reserved for the cloud shadow, snow/ice and cirrus
        fields, so treating those as a three-level scale reads a level that
        is not defined. :func:`qa_pixel_confidence` returns the raw value
        either way rather than inventing a meaning for it.
    HIGH : int
        3 — high confidence. This is the value each single-bit flag reports.

    Examples
    --------
    >>> QAConfidence.HIGH
    <QAConfidence.HIGH: 3>
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


#: The confidence fields for which ``10`` means Medium rather than Reserved.
#: Only cloud confidence defines it.
_QA_FIELDS_WITH_MEDIUM = frozenset({QAConfidenceField.CLOUD})

#: Bits that Landsat 4, 5 and 7 leave Unused, because TM and ETM+ have no
#: cirrus band to populate them from.
_QA_OLI_ONLY_FLAGS = frozenset({QAPixelFlag.CIRRUS})
_QA_OLI_ONLY_FIELDS = frozenset({QAConfidenceField.CIRRUS})

#: Landsat missions whose QA_PIXEL carries the cirrus bit and its confidence.
_QA_OLI_MISSIONS = frozenset({8, 9})

#: Missions with a documented QA_PIXEL bit index, mirroring the band tables.
_QA_MISSIONS = frozenset({4, 5, 7, 8, 9})

#: The flags Easy-EO treats as cloud by default: cloud, its dilation buffer,
#: cloud shadow, and cirrus where the sensor has it. Dilated cloud is included
#: because the plain cloud bit is a high-confidence claim and so stops short of
#: the cloud's own edge; cirrus for the reason :data:`SCL_CLOUDY` includes it.
QA_PIXEL_CLOUDY: frozenset[QAPixelFlag] = frozenset(
    {
        QAPixelFlag.DILATED_CLOUD,
        QAPixelFlag.CLOUD,
        QAPixelFlag.CLOUD_SHADOW,
        QAPixelFlag.CIRRUS,
    }
)

#: The flag that marks a pixel with no image data behind it at all.
QA_PIXEL_NODATA: frozenset[QAPixelFlag] = frozenset({QAPixelFlag.FILL})

#: What :func:`qa_pixel_mask` masks when not told otherwise.
QA_PIXEL_DEFAULT_MASKED: frozenset[QAPixelFlag] = QA_PIXEL_CLOUDY | QA_PIXEL_NODATA

#: The cloud-confidence level :func:`qa_pixel_mask` masks from by default.
#:
#: The single-bit cloud flag is set only where confidence is *High*, so a mask
#: built from the flags alone passes medium-confidence cloud through untouched
#: — USGS's own value table lists 22080 as "Mid conf cloud" with no flag set.
#: Masking from bits 8-9 instead catches it, which is why USGS says the
#: confidence fields, not the clear/cloud bits, are the truer measure of cloud
#: extent. Medium also matches what :data:`SCL_CLOUDY` does on Sentinel-2, so
#: one scene is not masked more leniently than another for no reason but the
#: satellite that took it.
#:
#: Only cloud confidence is thresholded this way: for cloud shadow and cirrus
#: the value 2 is Reserved rather than Medium, so ``>= High`` is the only
#: threshold above Low that exists, and that is exactly what their flag bits
#: already report.
QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE: QAConfidence = QAConfidence.MEDIUM


def _check_qa_mission(mission: int) -> int:
    """Validate a Landsat mission number against the documented bit indices."""
    try:
        number = operator.index(mission)
    except TypeError:
        raise ValidationError(
            f"mission must be a Landsat mission number, such as 9; got {mission!r}"
        ) from None
    if number not in _QA_MISSIONS:
        raise ValidationError(
            f"no QA_PIXEL bit index for Landsat {number}; the documented missions are "
            f"{', '.join(str(m) for m in sorted(_QA_MISSIONS))}"
        )
    return number


def qa_pixel_flags(mission: int) -> frozenset[QAPixelFlag]:
    """Return the ``QA_PIXEL`` flags one Landsat mission actually populates.

    The bit index is not the same on every mission: TM and ETM+ have no
    cirrus band, so Landsat 4, 5 and 7 document bit 2 as Unused where Landsat
    8 and 9 document it as cirrus. Reading it anyway would report a constant
    ``False`` as though it were a measurement.

    Parameters
    ----------
    mission : int
        Landsat mission number, e.g. ``9``.

    Returns
    -------
    frozenset of QAPixelFlag
        The flags this mission defines.

    Raises
    ------
    ValidationError
        If the mission has no documented ``QA_PIXEL`` bit index.

    Examples
    --------
    >>> QAPixelFlag.CIRRUS in qa_pixel_flags(9)
    True
    >>> QAPixelFlag.CIRRUS in qa_pixel_flags(7)
    False
    """
    if _check_qa_mission(mission) in _QA_OLI_MISSIONS:
        return frozenset(QAPixelFlag)
    return frozenset(QAPixelFlag) - _QA_OLI_ONLY_FLAGS


def qa_pixel_confidence_fields(mission: int) -> frozenset[QAConfidenceField]:
    """Return the ``QA_PIXEL`` confidence fields one Landsat mission populates.

    Parameters
    ----------
    mission : int
        Landsat mission number, e.g. ``9``.

    Returns
    -------
    frozenset of QAConfidenceField
        The confidence fields this mission defines. Bits 14-15 are Unused on
        Landsat 4, 5 and 7, so cirrus confidence is absent there.

    Raises
    ------
    ValidationError
        If the mission has no documented ``QA_PIXEL`` bit index.

    Examples
    --------
    >>> sorted(qa_pixel_confidence_fields(7))
    [<QAConfidenceField.CLOUD: 8>, <QAConfidenceField.CLOUD_SHADOW: 10>, \
<QAConfidenceField.SNOW_ICE: 12>]
    """
    if _check_qa_mission(mission) in _QA_OLI_MISSIONS:
        return frozenset(QAConfidenceField)
    return frozenset(QAConfidenceField) - _QA_OLI_ONLY_FIELDS


def confidence_has_medium(field: QAConfidenceField) -> bool:
    """Report whether ``10`` means Medium in this confidence field.

    Only cloud confidence defines the value; USGS documents it as Reserved
    for cloud shadow, snow/ice and cirrus. A caller writing "medium or above"
    logic needs to know which of the four it is looking at.

    Parameters
    ----------
    field : QAConfidenceField
        The confidence field.

    Returns
    -------
    bool
        True only for :attr:`QAConfidenceField.CLOUD`.

    Examples
    --------
    >>> confidence_has_medium(QAConfidenceField.CLOUD)
    True
    >>> confidence_has_medium(QAConfidenceField.CIRRUS)
    False
    """
    return QAConfidenceField(field) in _QA_FIELDS_WITH_MEDIUM


def _as_bitfield(qa: np.ndarray) -> np.ndarray:
    """Return ``qa`` in an integer dtype, so its bits can be read.

    ``QA_PIXEL`` is uint16, but stacking it beside float reflectance bands
    widens the whole stack to float, and a float cannot be shifted or masked.
    Every uint16 value is exactly representable in float32, so converting back
    is lossless — provided the values themselves were never interpolated.
    """
    dtype = getattr(qa, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        return qa
    if hasattr(qa, "astype"):
        return qa.astype("uint16")
    return np.asarray(qa, dtype="uint16")


def _resolve_qa_flag(value: QAPixelFlag | int | str) -> QAPixelFlag:
    """Resolve one flag name or bit position to a :class:`QAPixelFlag`."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in QAPixelFlag.__members__:
            return QAPixelFlag[name]
        raise ValidationError(
            f"no QA_PIXEL flag named {value!r}; the flags are "
            f"{', '.join(member.name.lower() for member in QAPixelFlag)}"
        )
    number = None
    if not isinstance(value, bool):
        try:
            number = operator.index(value)
        except TypeError:
            number = None
    if number is None:
        raise ValidationError(
            f"a QA_PIXEL flag must be a QAPixelFlag, a bit position 0-7, or a flag "
            f"name such as 'cloud_shadow'; got {value!r}"
        )
    try:
        return QAPixelFlag(number)
    except ValueError:
        raise ValidationError(
            f"bit {number} is not a QA_PIXEL flag; the single-bit flags are at "
            f"{', '.join(f'{m.value} {m.name.lower()}' for m in QAPixelFlag)}. Bits "
            f"8-15 are the paired confidence fields, read with qa_pixel_confidence"
        ) from None


def _resolve_qa_confidence_field(value: QAConfidenceField | int | str) -> QAConfidenceField:
    """Resolve one field name or low bit to a :class:`QAConfidenceField`."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in QAConfidenceField.__members__:
            return QAConfidenceField[name]
        raise ValidationError(
            f"no QA_PIXEL confidence field named {value!r}; the fields are "
            f"{', '.join(member.name.lower() for member in QAConfidenceField)}"
        )
    number = None
    if not isinstance(value, bool):
        try:
            number = operator.index(value)
        except TypeError:
            number = None
    if number is None:
        raise ValidationError(
            f"a confidence field must be a QAConfidenceField, its low bit (8, 10, 12 "
            f"or 14), or its name such as 'cloud_shadow'; got {value!r}"
        )
    try:
        return QAConfidenceField(number)
    except ValueError:
        raise ValidationError(
            f"bit {number} does not start a QA_PIXEL confidence field; the fields "
            f"begin at "
            f"{', '.join(f'{m.value} ({m.name.lower()})' for m in QAConfidenceField)}"
        ) from None


def resolve_qa_pixel_flags(
    flags: Iterable[QAPixelFlag | int | str],
) -> frozenset[QAPixelFlag]:
    """Resolve an iterable of flag names or bit positions to :class:`QAPixelFlag`.

    Parameters
    ----------
    flags : iterable of QAPixelFlag or int or str
        Flags, each given as a member, as its bit position (0-7), or as its
        name, matched case-insensitively (``"cloud_shadow"``).

    Returns
    -------
    frozenset of QAPixelFlag
        The resolved flags, with duplicates collapsed.

    Raises
    ------
    ValidationError
        If ``flags`` is a bare string rather than a sequence of them, if it is
        empty, or if any entry names no flag. A bit position of 8 or more is
        rejected by name: those bits are the paired confidence fields, and
        reading one of their bits alone would answer half a question.

    Examples
    --------
    >>> sorted(resolve_qa_pixel_flags(["cloud", 4]))
    [<QAPixelFlag.CLOUD: 3>, <QAPixelFlag.CLOUD_SHADOW: 4>]
    """
    if isinstance(flags, str):
        raise ValidationError(
            f"expected a sequence of QA_PIXEL flags, such as ['cloud', 'cloud_shadow']; "
            f"got {flags!r}"
        )
    resolved = frozenset(_resolve_qa_flag(value) for value in flags)
    if not resolved:
        raise ValidationError(
            "name at least one QA_PIXEL flag to mask; got an empty sequence. To mask "
            "nothing, skip the masking step rather than asking for an empty mask"
        )
    return resolved


def qa_pixel_flag(qa: np.ndarray, flag: QAPixelFlag | int | str, *, mission: int) -> np.ndarray:
    """Read one single-bit flag out of a Landsat ``QA_PIXEL`` band.

    Parameters
    ----------
    qa : numpy.ndarray
        The ``QA_PIXEL`` band. Values must have arrived by nearest neighbour:
        an interpolated bit field is not a bit field. A float copy is read
        back as uint16, which is lossless for the band's real values.
    flag : QAPixelFlag or int or str
        The flag to read, as a member, its bit position (0-7), or its name.
    mission : int
        Landsat mission number, e.g. ``9``. Required, because the same bit
        means different things across missions: bit 2 is cirrus on Landsat 8
        and 9 and Unused on Landsat 4, 5 and 7.

    Returns
    -------
    numpy.ndarray
        A boolean array of ``qa``'s shape, True where the flag is set.

    Raises
    ------
    ValidationError
        If ``flag`` names no flag, if ``mission`` has no documented bit index,
        or if the flag is one this mission leaves Unused.

    Notes
    -----
    Elementwise and array-agnostic: a lazy array yields a lazy result.

    Examples
    --------
    >>> import numpy as np
    >>> qa = np.array([1, 1 << 3, 0], dtype="uint16")
    >>> qa_pixel_flag(qa, "fill", mission=9)
    array([ True, False, False])
    >>> qa_pixel_flag(qa, QAPixelFlag.CLOUD, mission=9)
    array([False,  True, False])
    """
    wanted = _resolve_qa_flag(flag)
    available = qa_pixel_flags(mission)
    if wanted not in available:
        raise ValidationError(
            f"Landsat {mission} leaves QA_PIXEL bit {wanted.value} "
            f"({wanted.name.lower()}) Unused — TM and ETM+ have no cirrus band — so "
            f"there is nothing to read there. It is available on Landsat "
            f"{', '.join(str(m) for m in sorted(_QA_OLI_MISSIONS))}"
        )
    return (_as_bitfield(qa) & (1 << wanted.value)) != 0


def qa_pixel_confidence(
    qa: np.ndarray, field: QAConfidenceField | int | str, *, mission: int
) -> np.ndarray:
    """Read one two-bit confidence field out of a Landsat ``QA_PIXEL`` band.

    USGS validates the band against these fields rather than against the
    single-bit flags, and recommends using one or the other but not both: the
    flags are derived from the fields, so combining them double-counts.

    Parameters
    ----------
    qa : numpy.ndarray
        The ``QA_PIXEL`` band, as for :func:`qa_pixel_flag`.
    field : QAConfidenceField or int or str
        The field to read, as a member, its low bit (8, 10, 12 or 14), or its
        name (``"cloud_shadow"``, ``"snow_ice"``).
    mission : int
        Landsat mission number, e.g. ``9``. Required for the same reason as in
        :func:`qa_pixel_flag`: bits 14-15 are cirrus confidence on Landsat 8
        and 9 and Unused on Landsat 4, 5 and 7.

    Returns
    -------
    numpy.ndarray
        A uint16 array of ``qa``'s shape holding the field's raw value, 0-3,
        comparable against :class:`QAConfidence`. The value ``2`` means Medium
        only for cloud confidence and is Reserved elsewhere, so the raw number
        is returned rather than a level this function has interpreted; see
        :func:`confidence_has_medium`.

    Raises
    ------
    ValidationError
        If ``field`` names no confidence field, if ``mission`` has no
        documented bit index, or if the field is Unused on this mission.

    Notes
    -----
    Elementwise and array-agnostic: a lazy array yields a lazy result.

    Examples
    --------
    >>> import numpy as np
    >>> qa = np.array([3 << 8, 1 << 8, 0], dtype="uint16")
    >>> qa_pixel_confidence(qa, "cloud", mission=9)
    array([3, 1, 0], dtype=uint16)
    >>> qa_pixel_confidence(qa, "cloud", mission=9) == QAConfidence.HIGH
    array([ True, False, False])
    """
    wanted = _resolve_qa_confidence_field(field)
    if wanted not in qa_pixel_confidence_fields(mission):
        raise ValidationError(
            f"Landsat {mission} leaves QA_PIXEL bits {wanted.value}-{wanted.value + 1} "
            f"({wanted.name.lower()} confidence) Unused — TM and ETM+ have no cirrus "
            f"band — so there is nothing to read there. It is available on Landsat "
            f"{', '.join(str(m) for m in sorted(_QA_OLI_MISSIONS))}"
        )
    return (_as_bitfield(qa) >> wanted.value) & 3


def _resolve_confidence(value: QAConfidence | int) -> QAConfidence:
    """Resolve a confidence level, given as a member, a number, or a name."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in QAConfidence.__members__:
            return QAConfidence[name]
        raise ValidationError(
            f"no confidence level named {value!r}; the levels are "
            f"{', '.join(member.name.lower() for member in QAConfidence)}"
        )
    number = None
    if not isinstance(value, bool):
        try:
            number = operator.index(value)
        except TypeError:
            number = None
    if number is None:
        raise ValidationError(
            f"a confidence level must be a QAConfidence, a value 0-3, or a level "
            f"name such as 'medium'; got {value!r}"
        )
    try:
        return QAConfidence(number)
    except ValueError:
        raise ValidationError(
            f"{number} is not a confidence level; the levels are "
            f"{', '.join(f'{m.value} {m.name.lower()}' for m in QAConfidence)}"
        ) from None


def qa_pixel_mask(
    qa: np.ndarray,
    *,
    mission: int,
    flags: Iterable[QAPixelFlag | int | str] | None = None,
    min_cloud_confidence: QAConfidence | int | None = QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE,
) -> np.ndarray:
    """Flag the Landsat pixels for which any of a set of ``QA_PIXEL`` bits is set.

    The counterpart of :func:`scl_mask` for Landsat, producing the same thing
    from a different encoding: a boolean array that is True where the pixel
    should be discarded.

    Parameters
    ----------
    qa : numpy.ndarray
        The ``QA_PIXEL`` band, as for :func:`qa_pixel_flag`.
    mission : int
        Landsat mission number, e.g. ``9``. Required.
    flags : iterable of QAPixelFlag or int or str, optional
        Which flags to mask on. Defaults to :data:`QA_PIXEL_DEFAULT_MASKED` —
        fill, cloud, its dilation buffer, cloud shadow, and cirrus. A flag
        this mission leaves Unused is dropped from the **default** silently,
        so that the default works on every mission; naming one explicitly is
        still an error, because a caller who asks for cirrus on Landsat 7 has
        misunderstood something and should be told.
    min_cloud_confidence : QAConfidence or int or None, optional
        Also mask any pixel whose cloud confidence (bits 8-9) is at least this
        level. Defaults to :data:`QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE`
        (Medium), which is the only way to catch medium-confidence cloud: the
        cloud flag fires at High only. Pass ``None`` to mask from the flags
        alone, or :attr:`QAConfidence.HIGH` to reproduce the flag's own
        threshold.

    Returns
    -------
    numpy.ndarray
        A boolean array of ``qa``'s shape, True where any requested flag is
        set. Carries no nodata of its own.

    Raises
    ------
    ValidationError
        If ``flags`` is a bare string, is empty, names no flag, or explicitly
        names a flag this mission leaves Unused; or if
        ``min_cloud_confidence`` is not one of the four levels.

    Notes
    -----
    Elementwise and array-agnostic: a lazy array yields a lazy result. The
    flag bits are combined into one mask value and read in a single pass
    rather than one pass per flag.

    Examples
    --------
    >>> import numpy as np
    >>> qa = np.array([0, 1, 1 << 3, 1 << 5], dtype="uint16")
    >>> qa_pixel_mask(qa, mission=9)
    array([False,  True,  True, False])
    >>> qa_pixel_mask(qa, mission=9, flags=["snow"], min_cloud_confidence=None)
    array([False, False, False,  True])

    Medium-confidence cloud carries no flag of its own, so it survives a
    flags-only mask and not the default (22080 is USGS's own "Mid conf cloud"):

    >>> mid = np.array([22080], dtype="uint16")
    >>> qa_pixel_mask(mid, mission=9, min_cloud_confidence=None)
    array([False])
    >>> qa_pixel_mask(mid, mission=9)
    array([ True])
    """
    available = qa_pixel_flags(mission)
    if flags is None:
        wanted = QA_PIXEL_DEFAULT_MASKED & available
    else:
        wanted = resolve_qa_pixel_flags(flags)
        missing = wanted - available
        if missing:
            unusable = sorted(missing, key=lambda f: f.value)
            raise ValidationError(
                f"Landsat {mission} leaves QA_PIXEL bit"
                f"{'s' if len(unusable) > 1 else ''} "
                f"{', '.join(f'{f.value} ({f.name.lower()})' for f in unusable)} "
                f"Unused — TM and ETM+ have no cirrus band — so masking on "
                f"{'them' if len(unusable) > 1 else 'it'} would mask nothing. "
                f"Omit the flag, or leave flags unset to take the default, which "
                f"drops it on this mission"
            )
    bits = 0
    for flag in wanted:
        bits |= 1 << flag.value
    field = _as_bitfield(qa)
    masked = (field & bits) != 0
    if min_cloud_confidence is None:
        return masked
    threshold = _resolve_confidence(min_cloud_confidence)
    cloud_confidence = (field >> QAConfidenceField.CLOUD.value) & 3
    return masked | (cloud_confidence >= threshold.value)
