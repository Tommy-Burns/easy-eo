"""Synthetic Sentinel-2 and Landsat products, built for the test suite.

The suite blocks sockets, so no real product can be downloaded into it. These
builders stand in for one: structurally faithful, small enough to write in
milliseconds, and deterministic. Every test that needs a product on disk gets
it from here, so there is one answer to "what does a product look like" rather
than one per test module that can drift out of agreement.

Faithful means the parts a reader actually depends on, checked against real
output rather than documentation:

* Sentinel-2 — a namespaced manifest root over an unnamespaced body, the
  ``band_id``/``bandId`` indirection between the offset list and the spectral
  list, single-digit bands spelled ``B1`` in the spectral list and ``B01`` in
  the image paths, a granule holding its own ``MTD_TL.xml``, and images in the
  ``R10m``/``R20m``/``R60m`` layout at the resolutions each band is really
  written at.
* Landsat — the same content in all three metadata syntaxes, a Level-1 record
  repeating the Level-2 key names with different values, a scene centre time
  with seven fractional digits, and the fill values USGS declares: 0 in the
  imagery and 1 in the quality rasters.

Two levels of fixture are offered per mission, because the readers come in two
layers. ``manifest_xml``/``landsat_groups`` and the ``write_*`` helpers build
**metadata only**, for testing the parsers — including deliberately damaged
variants. ``build_safe``/``build_landsat`` build a **whole product** with real
images, for testing the loaders end to end.

Nothing here is imported by the library. ``scripts/make_product_fixture.py``
materialises one of these products into a directory you name, for looking at in
a GIS or stepping through by hand.
"""

from __future__ import annotations

import json
import tarfile
import zipfile

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin

# Sentinel-2

S2_PRODUCT = "S2B_MSIL2A_20240830T100559_N0511_R022_T32TPS_20240830T134009.SAFE"
S2_GRANULE = "L2A_T32TPS_A038765_20240830T100558"

#: The stem every image filename in this product starts with.
S2_STEM = "T32TPS_20240830T100559"

#: One image path, as the manifest writes them: relative to the .SAFE root and
#: without a file extension.
S2_IMAGE = f"GRANULE/{S2_GRANULE}/IMG_DATA/R10m/{S2_STEM}_B04_10m"

S2_CRS = "EPSG:32632"
S2_ULX, S2_ULY = 300000.0, 5100000.0

#: Width and height of the 10 m bands. Coarser bands are proportionally
#: smaller, as they are in a real product.
S2_SIZE_10M = 64

#: Which bands this product holds and at which resolutions, mirroring a real
#: one: B08 only at 10 m, B01 at 20 m and 60 m though it is sensed at 60, B09
#: only at 60 m, SCL at 20 m and 60 m.
S2_LAYOUT = {
    "B02": (10, 20),
    "B03": (10, 20),
    "B04": (10, 20, 60),
    "B08": (10,),
    "B05": (20, 60),
    "B11": (20, 60),
    "B12": (20, 60),
    "B8A": (20, 60),
    "B01": (20, 60),
    "B09": (60,),
    "SCL": (20, 60),
}

#: The digital number filling each band. Distinct per band, so a band read in
#: the wrong order shows up as a wrong value rather than a passing test.
S2_FILL = {
    "B02": 700,
    "B03": 900,
    "B04": 1234,
    "B08": 4321,
    "B05": 3000,
    "B11": 2222,
    "B12": 1800,
    "B8A": 4000,
    "B01": 1100,
    "B09": 1500,
}

# Spectral_Information spells single-digit bands "B1"; the image files use "B01".
S2_SPECTRAL = """
      <Spectral_Information_List>
        <Spectral_Information bandId="0" physicalBand="B1"/>
        <Spectral_Information bandId="3" physicalBand="B4"/>
        <Spectral_Information bandId="8" physicalBand="B8A"/>
        <Spectral_Information bandId="12" physicalBand="B12"/>
      </Spectral_Information_List>"""

S2_OFFSETS = """
      <BOA_ADD_OFFSET_VALUES_LIST>
        <BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET>
      </BOA_ADD_OFFSET_VALUES_LIST>"""

S2_TILE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_Tile_ID
    xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd">
  <n1:General_Info>
    <TILE_ID>S2B_OPER_MSI_L2A_TL_2BPS_20240830T134009_A038765_T32TPS_N05.11</TILE_ID>
    <SENSING_TIME>2024-08-30T10:06:21.919283Z</SENSING_TIME>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_NAME>WGS84 / UTM zone 32N</HORIZONTAL_CS_NAME>
      <HORIZONTAL_CS_CODE>EPSG:32632</HORIZONTAL_CS_CODE>
      <Size resolution="10"><NROWS>10980</NROWS><NCOLS>10980</NCOLS></Size>
    </Tile_Geocoding>
  </n1:Geometric_Info>
</n1:Level-2A_Tile_ID>
"""

#: The tile manifest's sensing time, which a load reports as its timestamp.
S2_SENSING_TIME = "2024-08-30T10:06:21.919283+00:00"


def manifest_xml(
    *,
    level="2A",
    product_type="S2MSI2A",
    baseline="05.11",
    quantification="10000",
    offsets=S2_OFFSETS,
    spectral=S2_SPECTRAL,
    start_time="2024-08-30T10:05:59.024Z",
    uri=S2_PRODUCT,
    granule=S2_GRANULE,
    image_files=(S2_IMAGE,),
):
    """Build a product manifest with the parts under test made adjustable."""
    type_element = f"<PRODUCT_TYPE>{product_type}</PRODUCT_TYPE>" if product_type else ""
    quant = (
        f"""
      <QUANTIFICATION_VALUES_LIST>
        <BOA_QUANTIFICATION_VALUE unit="none">{quantification}</BOA_QUANTIFICATION_VALUE>
      </QUANTIFICATION_VALUES_LIST>"""
        if quantification
        else ""
    )
    files = "".join(f"<IMAGE_FILE>{path}</IMAGE_FILE>" for path in image_files)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-{level}_User_Product
    xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-{level}.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>{start_time}</PRODUCT_START_TIME>
      <PRODUCT_URI>{uri}</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-{level}</PROCESSING_LEVEL>
      {type_element}
      <PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>
      <Product_Organisation>
        <Granule_List>
          <Granule granuleIdentifier="{granule}" imageFormat="JPEG2000">
            {files}
          </Granule>
        </Granule_List>
      </Product_Organisation>
    </Product_Info>
    <Product_Image_Characteristics>{quant}{offsets}{spectral}
    </Product_Image_Characteristics>
  </n1:General_Info>
</n1:Level-{level}_User_Product>
"""


def write_safe(root, *, manifest_name="MTD_MSIL2A.xml", tile_xml=S2_TILE_XML, **kwargs):
    """Write a .SAFE tree of metadata only, and return its directory.

    No image is written: the metadata reader never opens one, so a parser test
    should not pay to encode any.
    """
    safe = root / S2_PRODUCT
    (safe / "GRANULE" / S2_GRANULE).mkdir(parents=True, exist_ok=True)
    (safe / manifest_name).write_text(manifest_xml(**kwargs))
    if tile_xml is not None:
        (safe / "GRANULE" / S2_GRANULE / "MTD_TL.xml").write_text(tile_xml)
    return safe


def write_jp2(path, *, resolution, size, band):
    """Write one georeferenced JPEG 2000 image, losslessly."""
    if band == "SCL":
        # Distinct class numbers in blocks, so any blending is detectable.
        data = np.zeros((size, size), dtype="uint16")
        data[: size // 2] = 4  # vegetation
        data[size // 2 :] = 9  # cloud high probability
        data = data[None]
    else:
        data = np.full((1, size, size), S2_FILL[band], dtype="uint16")
    # Only the options JP2OpenJPEG accepts: copying a GeoTIFF profile would
    # carry INTERLEAVE and tiling keys the driver rejects.
    #
    # REVERSIBLE and QUALITY are both required, and only together: the writer
    # is lossy by default and stays lossy with either one alone, which quietly
    # rewrites a class number 4 as 3 and would make a resampling test read as a
    # loader bug. Real products are losslessly encoded, so the fixture must be
    # too or it is testing the compressor rather than the reader.
    profile = {
        "driver": "JP2OpenJPEG",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "uint16",
        "crs": S2_CRS,
        "transform": from_origin(S2_ULX, S2_ULY, resolution, resolution),
        "nodata": 0,
        "REVERSIBLE": "YES",
        "QUALITY": "100",
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def build_safe(root, *, level="2A", product_type="S2MSI2A", layout=None, omit_file=None):
    """Build a whole .SAFE product, real images included, and return its path.

    Parameters
    ----------
    root : Path
        Directory to build the ``.SAFE`` inside.
    level, product_type : str
        The level to write into the manifest's name and body. They are set
        separately so a product can be made to disagree with itself.
    layout : dict or None
        Band to the resolutions it is written at, defaulting to
        :data:`S2_LAYOUT`.
    omit_file : tuple or None
        A ``(band, resolution)`` the manifest lists but the tree does not hold
        — the state a truncated download leaves behind.
    """
    layout = S2_LAYOUT if layout is None else layout
    safe = root / S2_PRODUCT
    images = safe / "GRANULE" / S2_GRANULE / "IMG_DATA"

    entries = []
    for band, resolutions in layout.items():
        for resolution in resolutions:
            (images / f"R{resolution}m").mkdir(parents=True, exist_ok=True)
            relative = (
                f"GRANULE/{S2_GRANULE}/IMG_DATA/R{resolution}m/{S2_STEM}_{band}_{resolution}m"
            )
            entries.append(relative)
            if omit_file == (band, resolution):
                continue
            write_jp2(
                safe / f"{relative}.jp2",
                resolution=resolution,
                size=S2_SIZE_10M * 10 // resolution,
                band=band,
            )

    (safe / f"MTD_MSIL{level}.xml").write_text(
        manifest_xml(level=level, product_type=product_type, image_files=entries)
    )
    (safe / "GRANULE" / S2_GRANULE / "MTD_TL.xml").write_text(S2_TILE_XML)
    return safe


# Landsat

L9_ID = "LC09_L2SP_193028_20260822_20260823_02_T1"
L7_ID = "LE07_L2SP_193028_20231024_20231119_02_T2"
L1_ID = "LC09_L1TP_193028_20260822_20260823_02_T1"

LANDSAT_CRS = "EPSG:32633"
LANDSAT_ULX, LANDSAT_ULY = 300000.0, 5100000.0

#: The single grid every Collection 2 Level-2 band is delivered on.
LANDSAT_RES = 30.0
LANDSAT_SIZE = 64

#: Bands an OLI/TIRS product holds, with the digital number filling each.
OLI_BANDS = {
    "SR_B1": 8000,
    "SR_B2": 8100,
    "SR_B3": 9000,
    "SR_B4": 10000,
    "SR_B5": 25000,
    "SR_B6": 15000,
    "SR_B7": 12000,
    "ST_B10": 44000,
    "QA_PIXEL": 21824,
    "QA_RADSAT": 0,
    "SR_QA_AEROSOL": 96,
}

#: Bands a TM/ETM+ product holds. The band numbers carry different wavelengths
#: from OLI's, which is the whole reason bands are requested by name.
TM_BANDS = {
    "SR_B1": 8100,
    "SR_B2": 9000,
    "SR_B3": 10000,
    "SR_B4": 25000,
    "SR_B5": 15000,
    "SR_B7": 12000,
    "ST_B6": 44000,
    "QA_PIXEL": 5440,
    "QA_RADSAT": 0,
    "SR_ATMOS_OPACITY": 120,
    "SR_CLOUD_QA": 0,
}

#: Bands whose fill value is 1 rather than 0, as USGS declares them.
LANDSAT_QUALITY_FILL = {"QA_PIXEL"}

#: The three file tokens a metadata-only fixture lists, enough to exercise
#: reflectance, temperature, and quality without writing eleven of them.
LANDSAT_DEFAULT_TOKENS = ("SR_B4", "ST_B10", "QA_PIXEL")


def landsat_groups(
    *,
    level="L2SP",
    product_id=L9_ID,
    row="028",
    mission_prefix=None,
    tokens=LANDSAT_DEFAULT_TOKENS,
    date="2026-08-22",
):
    """Describe a Landsat product as the groups of key-value pairs it holds.

    The Level-1 groups are the point of the fixture as much as the Level-2 ones
    are: they repeat the same key names with different values, so a reader that
    searches for a bare key name passes or fails here by accident of ordering.
    """
    if mission_prefix:
        product_id = mission_prefix + product_id[4:]
    return {
        "PRODUCT_CONTENTS": {
            "LANDSAT_PRODUCT_ID": product_id,
            "PROCESSING_LEVEL": level,
            "COLLECTION_NUMBER": "02",
            "COLLECTION_CATEGORY": product_id[-2:],
            **{f"FILE_NAME_{token}": f"{product_id}_{token}.TIF" for token in tokens},
            "FILE_NAME_METADATA_ODL": f"{product_id}_MTL.txt",
        },
        "IMAGE_ATTRIBUTES": {
            "SPACECRAFT_ID": f"LANDSAT_{product_id[3]}",
            "SENSOR_ID": "OLI_TIRS",
            "WRS_PATH": "193",
            "WRS_ROW": row,
            "DATE_ACQUIRED": date,
            # USGS writes seven fractional digits.
            "SCENE_CENTER_TIME": "10:04:22.1183580Z",
        },
        "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS": {
            "REFLECTANCE_MULT_BAND_4": "2.75e-05",
            "REFLECTANCE_ADD_BAND_4": "-0.2",
        },
        "LEVEL2_SURFACE_TEMPERATURE_PARAMETERS": {
            "TEMPERATURE_MULT_BAND_ST_B10": "0.00341802",
            "TEMPERATURE_ADD_BAND_ST_B10": "149.0",
        },
        # The Level-1 record repeats the same key names with different values.
        # Everything above must win over everything here.
        "LEVEL1_PROCESSING_RECORD": {
            "LANDSAT_PRODUCT_ID": L1_ID,
            "PROCESSING_LEVEL": "L1TP",
        },
        "LEVEL1_RADIOMETRIC_RESCALING": {
            "REFLECTANCE_MULT_BAND_4": "2.0000E-05",
            "REFLECTANCE_ADD_BAND_4": "-0.100000",
        },
    }


def as_odl(groups):
    """Render groups in the ODL text syntax."""
    lines = ["GROUP = LANDSAT_METADATA_FILE"]
    for group, members in groups.items():
        lines.append(f"  GROUP = {group}")
        lines += [f'    {key} = "{value}"' for key, value in members.items()]
        lines.append(f"  END_GROUP = {group}")
    lines += ["END_GROUP = LANDSAT_METADATA_FILE", "END"]
    return "\n".join(lines)


def as_json(groups):
    """Render groups in the JSON syntax."""
    return json.dumps({"LANDSAT_METADATA_FILE": groups}, indent=2)


def as_xml(groups):
    """Render groups in the XML syntax."""
    body = "".join(
        f"<{group}>"
        + "".join(f"<{key}>{value}</{key}>" for key, value in members.items())
        + f"</{group}>"
        for group, members in groups.items()
    )
    return f"<LANDSAT_METADATA_FILE>{body}</LANDSAT_METADATA_FILE>"


RENDERERS = {"xml": as_xml, "json": as_json, "txt": as_odl}


def write_landsat_metadata(root, syntax="xml", **kwargs):
    """Write a product directory holding one metadata file and no imagery."""
    groups = landsat_groups(**kwargs)
    name = groups["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
    directory = root / name
    directory.mkdir(exist_ok=True)
    (directory / f"{name}_MTL.{syntax}").write_text(RENDERERS[syntax](groups))
    return directory


def write_landsat_tif(path, token, table):
    """Write one georeferenced 30 m GeoTIFF for a band."""
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=LANDSAT_SIZE,
        width=LANDSAT_SIZE,
        count=1,
        dtype="uint16",
        crs=LANDSAT_CRS,
        transform=from_origin(LANDSAT_ULX, LANDSAT_ULY, LANDSAT_RES, LANDSAT_RES),
        nodata=1 if token in LANDSAT_QUALITY_FILL else 0,
    ) as dst:
        dst.write(np.full((1, LANDSAT_SIZE, LANDSAT_SIZE), table[token], dtype="uint16"))


def landsat_table(product_id):
    """The band table a product's mission uses."""
    return OLI_BANDS if product_id.startswith("LC0") else TM_BANDS


def build_landsat(
    root, *, product_id=L9_ID, tokens=None, level="L2SP", syntax="json", omit_file=None
):
    """Build a whole Landsat product, real images included, and return its path.

    Parameters
    ----------
    root : Path
        Directory to build the product inside.
    product_id : str
        Which mission and scene to build. The prefix chooses the band table.
    tokens : sequence of str or None
        File tokens to write, defaulting to every band the mission holds.
        Dropping the ``ST_`` tokens makes a reflectance-only ``L2SR`` product.
    level : str
        Processing level to record, e.g. ``"L2SR"`` or ``"L1TP"``.
    syntax : str
        Which of the three metadata syntaxes to write.
    omit_file : str or None
        A token the metadata lists but the directory does not hold.
    """
    table = landsat_table(product_id)
    if tokens is None:
        tokens = list(table)
    directory = root / product_id
    directory.mkdir()
    date = "2026-08-22" if product_id.startswith("LC09") else "2023-10-24"
    groups = landsat_groups(product_id=product_id, tokens=tokens, level=level, date=date)
    (directory / f"{product_id}_MTL.{syntax}").write_text(RENDERERS[syntax](groups))
    for token in tokens:
        if token != omit_file:
            write_landsat_tif(directory / f"{product_id}_{token}.TIF", token, table)
    return directory


# Packing a built product into the archive it would have arrived in


def pack_zip(tree, archive, *, base):
    """Zip a directory tree, naming members relative to ``base``."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(base).as_posix())
    return archive


def pack_tar(tree, archive, *, base, compress=False):
    """Tar a directory tree, naming members relative to ``base``."""
    with tarfile.open(archive, "w:gz" if compress else "w") as tarred:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                tarred.add(path, arcname=path.relative_to(base).as_posix())
    return archive
