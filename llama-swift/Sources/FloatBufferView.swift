import Foundation

/// FloatBufferView
public struct FloatBufferView: RandomAccessCollection {
  public typealias Element = Float
  public typealias Index = Int

  /// 元データ
  private let data: Data
  /// このビューが始まるData内のオフセット
  private let baseOffset: Data.Index
  /// このビューに含まれるFloatの数
  public let count: Int

  public var startIndex: Index { 0 }
  public var endIndex: Index { count }

  /// Initializer
  /// - Parameters:
  ///  - data: 元データ
  ///  - range: この・ビューが対象とするDataのバイト範囲
  init(data: Data, range: Range<Data.Index>) {
    let floatStride = MemoryLayout<Float>.stride
    precondition(
      range.lowerBound >= data.startIndex && range.upperBound <= data.endIndex, "Invalid range")
    precondition(range.count % floatStride == 0, "Range must be a multiple of \(floatStride) bytes")

    self.data = data
    self.baseOffset = range.lowerBound
    self.count = range.count / floatStride
  }

  /// 添字アクセス
  public subscript(position: Index) -> Element {
    precondition(indices.contains(position), "Index out of bounds")
    let byteOffset = baseOffset + position * MemoryLayout<Float>.stride

    // 個々の要素アクセスでも安全のため withUnsafeBytes を使う
    return data.withUnsafeBytes { rawBufferPointer in
      // .loadUnaligned はアラインメントを気にしないが、パフォーマンスに影響する場合がある
      let uint32Value = rawBufferPointer.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)
      // リトルエンディアン前提のデータであるため
      return Float(bitPattern: uint32Value.littleEndian)
    }
  }

  /// 内部のUnsafeBufferPointerに安全にアクセスするためのメソッド
  public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows
    -> R
  {
    let floatStride = MemoryLayout<Float>.stride
    let range = baseOffset..<baseOffset + count * floatStride

    return try data.withUnsafeBytes { rawBufferPointer in
      let slicePointer = UnsafeRawBufferPointer(rebasing: rawBufferPointer[range])
      guard let baseAddress = slicePointer.baseAddress else {
        return try body(UnsafeBufferPointer<Float>(start: nil, count: 0))
      }
      let floatPointer = baseAddress.assumingMemoryBound(to: Float.self)
      let bufferPointer = UnsafeBufferPointer<Float>(start: floatPointer, count: self.count)
      return try body(bufferPointer)
    }
  }
}
